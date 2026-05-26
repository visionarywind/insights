"""DeepSeekV4 key source excerpts with numeric comments.

This file is an annotated reading copy, not a model implementation.

Source basis:
    repos/transformers/src/transformers/models/deepseek_v4/configuration_deepseek_v4.py
    repos/transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py
    repos/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py

Purpose:
    The real source contains framework code, tensor ops, caches, and generated code.
    This file keeps only the key statements and adds concrete numeric comments next
    to them. Use it together with run_deepseek_v4_tiny_trace.py.
"""


# =============================================================================
# 1. DeepseekV4Config.__post_init__
# =============================================================================

# Real source pattern:
#
# if self.layer_types is None:
#     interleave = [
#         "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
#         for i in range(max(n - 2, 0))
#     ]
#     self.layer_types = ["heavily_compressed_attention"] * min(n, 2) + interleave
#
# Numeric example:
#   n = 4
#   range(max(4 - 2, 0)) = range(2), so i = 0, 1
#   i = 0 -> heavily_compressed_attention
#   i = 1 -> compressed_sparse_attention
#
# Final layer_types:
#   layer0 = heavily_compressed_attention
#   layer1 = heavily_compressed_attention
#   layer2 = heavily_compressed_attention
#   layer3 = compressed_sparse_attention
#
# Meaning:
#   layer_types[i] decides which attention path layer i uses.


# Real source pattern:
#
# if self.mlp_layer_types is None:
#     n_hash = legacy_num_hash_layers if legacy_num_hash_layers is not None else self.default_num_hash_layers
#     self.mlp_layer_types = ["hash_moe"] * min(n, n_hash) + ["moe"] * max(0, n - n_hash)
#
# Numeric example:
#   n = 5
#   n_hash = 3
#   ["hash_moe"] * 3 + ["moe"] * 2
#
# Final mlp_layer_types:
#   layer0 = hash_moe
#   layer1 = hash_moe
#   layer2 = hash_moe
#   layer3 = moe
#   layer4 = moe


# =============================================================================
# 2. DeepseekV4Model.forward: embedding -> mHC streams
# =============================================================================

# Real source pattern:
#
# inputs_embeds = self.embed_tokens(input_ids)
#
# Numeric example:
#   input_ids = [1, 2, 3, 4]
#   embedding_table[1] = [1, 0]
#   embedding_table[2] = [0, 1]
#   embedding_table[3] = [1, 1]
#   embedding_table[4] = [2, 1]
#
# Result:
#   inputs_embeds =
#   token0 [1, 0]
#   token1 [0, 1]
#   token2 [1, 1]
#   token3 [2, 1]
#
# Shape:
#   input_ids.shape     = [1, 4]
#   inputs_embeds.shape = [1, 4, hidden_size]


# Real source pattern:
#
# hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
#
# Numeric example with hc_mult = 2:
#   token0 hidden = [1, 0]
#   token0 stream0 = [1, 0]
#   token0 stream1 = [1, 0]
#
# In run_deepseek_v4_tiny_trace.py, stream1 is set to hidden + [0.1, -0.1]
# only to make the two streams visibly different:
#   token0 stream0 = [1, 0]
#   token0 stream1 = [1.1, -0.1]
#
# Shape:
#   before = [B, S, D]
#   after  = [B, S, hc_mult, D]


# Real source pattern:
#
# position_embeddings = {
#     "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
#     "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
# }
#
# Meaning:
#   main     -> sliding attention path
#   compress -> HCA, CSA, and CSA indexer path


# =============================================================================
# 3. DeepseekV4DecoderLayer.forward: one layer structure
# =============================================================================

# Real source pattern:
#
# post, comb, collapsed = self.attn_hc(hidden_states)
# attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
# hidden_states = post.unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), hidden_states)
#
# Meaning:
#   1. hidden_states has multiple mHC streams: [B, S, hc_mult, D]
#   2. attn_hc collapses them into one attention input: [B, S, D]
#   3. attention computes attn_output: [B, S, D]
#   4. post writes attention output back to each stream
#   5. comb mixes the previous streams
#
# Numeric example for one token:
#   old stream0 = [1.0]
#   old stream1 = [2.0]
#   attn_output = [10.0]
#   post = [0.3, 0.7]
#   comb =
#     [[0.8, 0.2],
#      [0.2, 0.8]]
#
# New stream0:
#   0.3 * 10.0 + 0.8 * 1.0 + 0.2 * 2.0
#   = 3.0 + 0.8 + 0.4
#   = 4.2
#
# New stream1:
#   0.7 * 10.0 + 0.2 * 1.0 + 0.8 * 2.0
#   = 7.0 + 0.2 + 1.6
#   = 8.8


# Real source pattern:
#
# post, comb, collapsed = self.ffn_hc(hidden_states)
# mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
# return post.unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), hidden_states)
#
# Meaning:
#   The MoE branch repeats the same pattern:
#   mHC collapse -> MoE -> mHC write back.


# =============================================================================
# 4. DeepseekV4HyperConnection.forward: collapse multiple streams
# =============================================================================

# Real source pattern:
#
# flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
# pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
# pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
# collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
#
# Meaning:
#   pre decides how to combine multiple streams into one vector.
#
# Numeric example:
#   stream0 = [1.0, 0.0]
#   stream1 = [3.0, 2.0]
#   pre = [0.25, 0.75]
#
# collapsed dim0:
#   0.25 * 1.0 + 0.75 * 3.0 = 2.5
#
# collapsed dim1:
#   0.25 * 0.0 + 0.75 * 2.0 = 1.5
#
# collapsed = [2.5, 1.5]


# =============================================================================
# 5. DeepseekV4Attention.forward: Q/KV and compressed KV
# =============================================================================

# Real source pattern:
#
# q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
# q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
# q = self.q_b_norm(q)
# q = apply_rotary_pos_emb(q, cos, sin)
#
# Shape example:
#   B = 1
#   S = 4
#   num_heads = 2
#   head_dim = 3
#
# q_b_proj output shape = [1, 4, 6]
# view                  = [1, 4, 2, 3]
# transpose             = [1, 2, 4, 3]
#
# Meaning:
#   2 heads, each head has 3 values per token.


# Real source pattern:
#
# kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
# kv = apply_rotary_pos_emb(kv, cos, sin)
#
# Shape example:
#   kv_proj output = [1, 4, 3]
#   view           = [1, 4, 1, 3]
#   transpose      = [1, 1, 4, 3]
#
# Meaning:
#   DeepSeekV4 uses a shared KV head. One KV head is broadcast to all query heads.


# Real source pattern:
#
# if self.compressor is not None:
#     compressed_kv, block_bias = self.compressor(...)
#     kv = torch.cat([kv, compressed_kv], dim=2)
#
# Numeric shape example:
#   sliding kv length    = 4
#   compressed kv length = 2
#
# Before cat:
#   kv.shape = [B, 1, 4, head_dim]
#
# After cat:
#   kv.shape = [B, 1, 6, head_dim]
#
# Meaning:
#   attention can read local sliding KV plus long-range compressed KV.


# =============================================================================
# 6. HCA: DeepseekV4HCACompressor.forward
# =============================================================================

# Real source pattern:
#
# kv = self.kv_proj(hidden_states)
# gate = self.gate_proj(hidden_states)
# chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
# chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias
# compressed = self.kv_norm((chunk_kv * chunk_gate.softmax(dim=2)).sum(dim=2))
#
# Meaning:
#   HCA groups tokens into fixed windows.
#   Each window becomes one compressed entry.
#   Real source uses gate softmax weights.
#   The trace script uses simple average to make the arithmetic visible.


# Trace script example:
#   compress_rate = 2
#   token0 = [1.274, -0.039]
#   token1 = [0.260,  0.906]
#   token2 = [0.859,  1.158]
#   token3 = [2.209,  0.993]
#
# HCA creates:
#   entry0 = avg(token0, token1)
#   entry1 = avg(token2, token3)
#
# entry0 dim0:
#   (1.274 + 0.260) / 2 = 0.767
# entry0 dim1:
#   (-0.039 + 0.906) / 2 = 0.433
# entry0:
#   [0.767, 0.433]
#
# entry1 dim0:
#   (0.859 + 2.209) / 2 = 1.534
# entry1 dim1:
#   (1.158 + 0.993) / 2 = 1.076
# entry1:
#   [1.534, 1.076]


# Real source pattern:
#
# causal_threshold = (position_ids + 1) // self.compress_rate
# block_bias = block_bias.masked_fill(entry_indices >= causal_threshold, float("-inf"))
#
# Trace script equivalent:
#   visible_count = (position + 1) // compress_rate
#   visible = compressed[:visible_count]
#
# With compress_rate = 2:
#
# token0:
#   visible_count = (0 + 1) // 2 = 0
#   visible entries = none
#
# token1:
#   visible_count = (1 + 1) // 2 = 1
#   visible entries = entry0
#
# token2:
#   visible_count = (2 + 1) // 2 = 1
#   visible entries = entry0
#
# token3:
#   visible_count = (3 + 1) // 2 = 2
#   visible entries = entry0, entry1
#
# Meaning:
#   A token can only see compressed entries that are already complete.


# Trace script local context example for token2:
#
# Real sliding attention uses QK^T -> softmax -> weighted sum(V).
# The script uses a local average.
#
# window_size = 2
# position = 2
# start = max(0, position - window_size + 1)
#       = max(0, 2 - 2 + 1)
#       = 1
#
# window = tokens 1-2
# token1 = [0.260, 0.906]
# token2 = [0.859, 1.158]
#
# local dim0:
#   (0.260 + 0.859) / 2 = 0.559
#
# local dim1:
#   (0.906 + 1.158) / 2 = 1.032
#
# local = [0.559, 1.032]
#
# token2 can see entry0:
#   entry0 = [0.767, 0.433]
#
# output dim0:
#   (local.dim0 + entry0.dim0) / 2
#   = (0.559 + 0.767) / 2
#   = 0.663
#
# output dim1:
#   (local.dim1 + entry0.dim1) / 2
#   = (1.032 + 0.433) / 2
#   = 0.733
#
# HCA output for token2:
#   [0.663, 0.733]


# =============================================================================
# 7. CSA: DeepseekV4CSACompressor.forward and DeepseekV4Indexer.forward
# =============================================================================

# Real source pattern:
#
# new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
# new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
#
# Meaning:
#   CSA splits each token projection into Ca and Cb.
#   Current window contributes Cb.
#   Previous window contributes Ca.
#
# Simplified trace script:
#   compress_rate = 2
#   entry0 = avg(token0, token1)
#   entry1 = avg(token2, token3)
#
# Real CSA is more complex than this average. The trace keeps the entry concept
# visible before explaining indexer selection.


# Real source pattern:
#
# scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
# scores = F.relu(scores) * self.softmax_scale
# index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)
# top_k_indices = index_scores.topk(top_k, dim=-1).indices
#
# Trace script example for token3:
#   query = [2.196, 0.993]
#   entry0 = [1.009, 0.407]
#   entry1 = [1.503, 1.077]
#
# score(entry0):
#   2.196 * 1.009 + 0.993 * 0.407 = 2.619
#
# score(entry1):
#   2.196 * 1.503 + 0.993 * 1.077 = 4.370
#
# selected entry:
#   argmax([2.619, 4.370]) = entry1
#
# Meaning:
#   CSA does not let every query use every compressed entry.
#   Indexer chooses the most relevant compressed entries.


# =============================================================================
# 8. MoE: DeepseekV4SparseMoeBlock, routers, experts
# =============================================================================

# Real source pattern:
#
# if self.is_hash:
#     _, weights, indices = self.gate(hidden_states, input_ids)
# else:
#     _, weights, indices = self.gate(hidden_states)
# routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
# return routed + self.shared_experts(residual)
#
# Meaning:
#   routed experts process selected tokens.
#   shared expert processes every token.
#   final MoE output is routed + shared.


# Trace script example for token0:
#   input = [x, y] = [1.0425, -0.0425]
#
# router_scores:
#   expert0 score = x = 1.0425
#   expert1 score = y = -0.0425
#   expert2 score = 0.5 * (x + y)
#                 = 0.5 * (1.0425 - 0.0425)
#                 = 0.5
#
# selected experts:
#   top2([1.0425, -0.0425, 0.5]) = [expert0, expert2]
#
# weights:
#   score_sum = 1.0425 + 0.5 = 1.5425
#   expert0 weight = 1.0425 / 1.5425 = 0.676
#   expert2 weight = 0.5 / 1.5425 = 0.324
#
# expert0 output:
#   [1.4 * x, 0.7 * y]
#   = [1.4 * 1.0425, 0.7 * -0.0425]
#   = [1.459, -0.030]
#
# expert2 output:
#   [1.1 * x, 1.1 * y]
#   = [1.1 * 1.0425, 1.1 * -0.0425]
#   = [1.147, -0.047]
#
# routed output:
#   dim0 = 0.676 * 1.459 + 0.324 * 1.147 = 1.358
#   dim1 = 0.676 * -0.030 + 0.324 * -0.047 = -0.035
#
# shared expert:
#   0.2 * input = [0.209, -0.009]
#
# MoE output:
#   [1.358, -0.035] + [0.209, -0.009]
#   = [1.567, -0.044]


# =============================================================================
# 9. DeepseekV4ForCausalLM.forward: lm_head
# =============================================================================

# Real source pattern:
#
# hidden_states = outputs.last_hidden_state
# logits = self.lm_head(hidden_states[:, slice_indices, :])
#
# Trace script example:
#   hidden = [x, y] = [2.275, 1.063]
#   score(A) = x
#   score(B) = y
#   score(C) = 0.3 * x + 0.7 * y
#
# score(A):
#   2.275
#
# score(B):
#   1.063
#
# score(C):
#   0.3 * 2.275 + 0.7 * 1.063 = 1.426
#
# logits:
#   [2.275, 1.063, 1.426]


if __name__ == "__main__":
    print("This file is an annotated source reading aid.")
    print("Run run_deepseek_v4_tiny_trace.py for executable numeric output.")
