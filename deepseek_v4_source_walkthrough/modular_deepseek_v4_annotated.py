# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# =============================================================================
# DeepSeekV4 源码行内注释版
# =============================================================================
#
# This file was copied from:
#   repos/transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py
#
# 本文件只放在 insights/ 下用于阅读源码，不作为 Transformers 的运行源码。
#
# 注释规则：
#   1. 保留原源码结构，不重写实现。
#   2. 在关键语句旁边补充中文解释。
#   3. 用 run_deepseek_v4_tiny_trace.py 中的 4-token 示例解释计算过程。
#   4. 真实模型使用矩阵乘、softmax、cache 和 RoPE；注释中的小数值例子只用于
#      说明数据如何流动，不表示真实权重输出。
#
# 最小示例全局设定：
#   input_ids = [1, 2, 3, 4]
#   hidden_dim = 2
#   hc_mult = 2
#   compress_rate = 2
#
# 在 HCA 示例中，进入 HCA attention 前的 hidden vector 为：
#   token0 = [1.274, -0.039]
#   token1 = [0.260,  0.906]
#   token2 = [0.859,  1.158]
#   token3 = [2.209,  0.993]
#
# HCA 会把 token0-token1 压成 entry0，把 token2-token3 压成 entry1：
#   entry0 = [(1.274 + 0.260)/2, (-0.039 + 0.906)/2] = [0.767, 0.433]
#   entry1 = [(0.859 + 2.209)/2, (1.158 + 0.993)/2] = [1.534, 1.076]
#
# 这些 entry 会作为 compressed KV 拼到普通 sliding KV 后面，供 attention 使用。
#
# 阅读路线：
#   DeepseekV4Model.forward
#     -> DeepseekV4DecoderLayer.forward
#     -> DeepseekV4Attention.forward
#     -> DeepseekV4HCACompressor.forward / DeepseekV4CSACompressor.forward
#     -> DeepseekV4SparseMoeBlock.forward
#     -> DeepseekV4HyperHead.forward
#
# 说明：
#   本文件保留源码结构。新增注释只解释“当前语句在做什么”和“最小数值例子如何算”。
#   例子中为了便于人工计算，常把线性层看成简单映射，把 softmax 权重设成平均权重。
#   真实模型会使用 checkpoint 权重，数值不会等于这些示例值，但张量流向和计算位置一致。
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, DynamicSlidingWindowLayer
from ...integrations import use_experts_implementation
from ...masking_utils import create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm
from ..glm.modeling_glm import rotate_half
from ..gpt_oss.modeling_gpt_oss import eager_attention_forward
from ..laguna.modeling_laguna import LagunaRotaryEmbedding
from ..llama.modeling_llama import LlamaMLP, LlamaModel
from ..mixtral.modeling_mixtral import MixtralExperts, MixtralForCausalLM, MixtralPreTrainedModel, MixtralTopKRouter
from .configuration_deepseek_v4 import DeepseekV4Config


logger = logging.get_logger(__name__)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """V4 interleaved RoPE applied to the *trailing* rope slice of `x`.

    `cos` / `sin` come in half-sized (one entry per interleaved pair, from
    `DeepseekV4RotaryEmbedding`); we expand them to the full rope dim with
    `repeat_interleave`, then rotate the last `2 * cos.shape[-1]` channels of `x`
    with the standard `x*cos + rotate_half(x)*sin` formula in fp32 and leave the
    leading nope channels untouched. V4-Flash lays each head out as `[nope | rope]`,
    matching the reference's `x[..., -rd:]` indexing.
    """
    # 输入 x 通常是 [B, num_heads, S, head_dim]。
    # head_dim 被拆成两段：
    #   nope：不做 RoPE 的前半段；
    #   rope：做 RoPE 的最后 rope_dim 个通道。
    #
    # DeepSeekV4 的 RoPE 是 interleaved 形式，cos/sin 只存每一对通道的一个角度。
    # 例如 cos=[0.8, 0.5]，repeat_interleave 后变成 [0.8, 0.8, 0.5, 0.5]，
    # 这样第 0/1 通道共用第一个角度，第 2/3 通道共用第二个角度。
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    # rotate_half 按相邻通道成对旋转：
    #   rope=[2, 4] -> rotate_half(rope)=[-4, 2]
    # 若 cos=[0.8,0.8]、sin=[0.6,0.6]：
    #   rotated_dim0 = 2*0.8 + (-4)*0.6 = -0.8
    #   rotated_dim1 = 4*0.8 + 2*0.6    =  4.4
    # 前面的 nope 不变，最后 cat 回完整 head。
    rotated = ((rope.float() * cos) + (rotate_half(rope).float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


class DeepseekV4RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV4UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm 只按最后一维做归一化，不减均值。
        # 示例：x=[3,4]
        #   square=[9,16]
        #   mean=12.5
        #   rsqrt(12.5)=0.283
        #   output=[3*0.283, 4*0.283]=[0.849, 1.131]
        # 这里没有可学习 weight，所以叫 UnweightedRMSNorm。
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class DeepseekV4RotaryEmbedding(LagunaRotaryEmbedding):
    """
    Multi-layer-type rotary embedding (Laguna pattern: partial rotary on top of
    Gemma3's per-layer-type buffers), specialised for V4's *interleaved* RoPE.
    Interleaved RoPE: one `θ_i` per pair (`rope_head_dim // 2` entries),
    DIFF no end-to-end duplication. Same shape as `inv_freq @ position_ids`.

    V4 deliberately decouples its architecture `layer_types`
    (`sliding_attention` / `compressed_sparse_attention` /
    `heavily_compressed_attention`) from its rope-type labels (`main` /
    `compress`) — the latter live as keys in `config.rope_parameters` and
    only differ in their `rope_theta` base. So this override replaces
    Laguna's `set(config.layer_types)` iteration with `rope_parameters.keys()`
    when building the per-type inv_freq buffers.
    """

    def __init__(self, config: DeepseekV4Config):
        nn.Module.__init__(self)
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        # Only the nested per-rope-type sub-dicts are real layer types — the top-level
        # `rope_type` key that ``convert_rope_params_to_dict`` may leave on
        # ``config.rope_parameters`` is a flat-shape leftover, not a layer.
        self.layer_types = [k for k, v in config.rope_parameters.items() if isinstance(v, dict)]
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = config.rope_parameters[layer_type]
            self.rope_type[layer_type] = rope_params["rope_type"]
            rope_init_fn = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            inv_freq, attention_scaling = rope_init_fn(config, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    def forward(self, x, position_ids, layer_type=None):
        # Key difference vs Laguna's forward: no `torch.cat([freqs, freqs], dim=-1)`
        # duplication. V4's interleaved RoPE pairs consecutive channels, so we only need
        # `rope_head_dim // 2` unique θ entries — the `apply_rotary_pos_emb` helper does
        # the `repeat_interleave(2)` next to the rotation math, where the link between
        # the doubled dim and `rotate_half` is local and obvious.
        # layer_type 只能是 "main" 或 "compress"。
        # "main" 给普通滑动窗口 attention 使用；
        # "compress" 给 HCA/CSA 的压缩 KV 和对应 query 使用。
        # 两者主要区别是 RoPE 的 theta 配置不同。
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            # 形状示例：
            #   position_ids=[0,1,2,3]，inv_freq 长度为 rope_dim/2。
            #   freqs 的形状是 [B, S, rope_dim/2]。
            #   cos/sin 先在这里保持半维度，真正扩成完整 rope_dim 的动作在
            #   apply_rotary_pos_emb() 里完成。
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            cos = freqs.cos() * attention_scaling
            sin = freqs.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DeepseekV4HCACache(DynamicSlidingWindowLayer):
    r"""Cache layer for HCA blocks (paper §2.3.2). Holds the long-range compressor's
    buffer / running compressed entries / count on top of the sliding-window K=V
    branch. HCA uses *non-overlapping* windows, so there is *no* overlap state,
    and HCA has *no* indexer either.

    State is dict-keyed by entry name — HCA only uses `"compressor"`, but
    :class:`DeepseekV4CSACache` adds `"indexer"` to the same dicts so a single
    set of methods (`store_compression_weights` / `update_compressor_states`)
    serves both:

      * `compressed_kv[name]` — the running list of compressed KV entries
        emitted so far (one every `compress_rate` source tokens; the long-range
        KVs the attention concatenates onto its sliding-window keys / values).
      * `buffer_kv[name]` / `buffer_gate[name]` — source tokens that arrived
        between two full windows; once the buffer hits `compress_rate` tokens
        the compressor closes a window, emits one entry, and drains the buffer.
      * `entry_count[name]` — number of compressed entries emitted so far, so
        `entry_count[name] * compress_rate` is the absolute position of the
        *next* window's first source token. Tracked separately from
        `position_ids` so prefill -> decode -> prefill stays consistent.
    """

    layer_type = "heavily_compressed_attention"

    def __init__(self, config: "DeepseekV4Config"):
        super().__init__(config)
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.buffer_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.buffer_gate: dict[str, torch.Tensor | None] = {"compressor": None}
        self.compressed_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.entry_count: dict[str, int] = {"compressor": 0}

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        """
        Shared sliding-window K=V update body. V4 uses shared-KV MQA, so `keys` and
        `values` point to the same storage on every layer.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.values = self.keys
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        self.keys = full[:, :, -self.sliding_window + 1 :, :]
        self.values = self.keys
        return full, full

    def store_compression_weights(
        self, name: str, kv: torch.Tensor, gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        r"""
        Concatenate the new projected `(kv, gate)` (paper §2.3.2 eqs. 20–21:
        `C = H·W^{KV}`, `Z = H·W^Z`) for entry `name` with what's already in
        the buffer, peel off the longest window-aligned prefix (the chunk
        ready to compress), keep the leftover in the buffer for next call,
        and return `(chunk_kv, chunk_gate, first_window_position)`. The
        returned chunk is softmax-aggregated by the compressor with
        `position_bias` to emit one compressed entry per window of
        `compress_rate` tokens.
        """
        first_window_position = self.entry_count[name] * self.compress_rate
        buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
        if buffered_kv is not None and buffered_kv.shape[1]:
            # decode 时可能一次只来 1 个 token。
            # 如果 compress_rate=2，上一次只来了 token0，就先存在 buffer。
            # 这一次来了 token1，把 buffer 中的 token0 和当前 token1 拼起来，
            # 才能形成一个完整窗口 token0-token1。
            kv = torch.cat([buffered_kv, kv], dim=1)
            gate = torch.cat([buffered_gate, gate], dim=1)
        # only return the longest prefix that's a multiple of compress_rate; the rest stays in the buffer for next time
        # usable 是当前可以压缩的 token 数。
        # 示例：compress_rate=2
        #   当前 kv 有 5 个 token -> usable=(5//2)*2=4
        #   前 4 个 token 压成 2 个 entry，最后 1 个 token 留到下一次 forward。
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        self.buffer_kv[name], self.buffer_gate[name] = kv[:, usable:], gate[:, usable:]
        return kv[:, :usable], gate[:, :usable], first_window_position

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        r"""
        Append freshly emitted compressed entries to `compressed_kv[name]`
        (`C^{Comp}`, paper §2.3.2 eq. 23), bump `entry_count[name]`, and
        return the running `compressed_kv[name]`.
        """
        if self.compressed_kv[name] is None:
            # 第一次产生 compressed entries，直接保存。
            # 示例：entry0=[0.767,0.433]，entry1=[1.534,1.076]
            # compressed_kv["compressor"] = [entry0, entry1]
            self.compressed_kv[name] = compressed
        elif compressed.shape[1] > 0:
            # 后续 decode/prefill 又产生新的 entry，就追加到历史后面。
            # attention 会把这些历史 entry 作为长程 KV 读取。
            self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=1)
        self.entry_count[name] += compressed.shape[1]
        return self.compressed_kv[name]


class DeepseekV4CSACache(DeepseekV4HCACache):
    r"""Cache layer for CSA blocks (paper §2.3.1). Extends :class:`DeepseekV4HCACache`
    by adding an `"indexer"` entry to the inherited `buffer_kv` / `buffer_gate` /
    `compressed_kv` / `entry_count` dicts, plus per-name *overlap* state for the
    two-series window scheme.

    What "overlap" means here: the CSA `kv_proj` / `gate_proj` produce `2 * head_dim`
    features per source token — two independent compressed series Ca and Cb stored
    in one tensor. Ca occupies `[..., :head_dim]`, Cb occupies `[..., head_dim:]`.
    Pooled entry `w` is the softmax-gated convex combination of window `w-1`'s Ca
    slice with window `w`'s Cb slice — effective width `2 * compress_rate_csa`,
    stride `compress_rate_csa` (paper §2.3.1).

    Because adjacent windows share state only through *the previous window's Ca
    slice*, the only thing we need to carry across a forward boundary is
    `chunk[:, -1, :, :head_dim]` (Ca) of the last full window — Cb is never read
    again. That's what `overlap_kv[name]` / `overlap_gate[name]` persist.
    """

    layer_type = "compressed_sparse_attention"

    def __init__(self, config: "DeepseekV4Config"):
        super().__init__(config)
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.buffer_kv["indexer"] = None
        self.buffer_gate["indexer"] = None
        self.compressed_kv["indexer"] = None
        self.entry_count["indexer"] = 0
        self.overlap_kv: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        self.overlap_gate: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}

    def update_overlap_state(
        self, name: str, chunk_kv: torch.Tensor, chunk_gate: torch.Tensor, head_dim: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        r"""
        Read the `name` entry's prior window's Ca slice (saved on the previous
        forward call) and persist the *current* call's last-window Ca slice for
        the next call. Only the `:head_dim` slice (Ca) is ever consumed
        downstream — Cb has already been folded into the previous window's
        emitted compressed entry — so we store half what `chunk[:, -1]` holds.
        Returns `(prior_kv, prior_gate)` — both `None` on the very first call.
        """
        prior_kv, prior_gate = self.overlap_kv[name], self.overlap_gate[name]
        # CSA 与 HCA 的区别：
        #   HCA 每个窗口独立压缩，token0-token1 -> entry0，token2-token3 -> entry1。
        #   CSA 使用重叠窗口，当前窗口的 Ca 会在下一个窗口中继续使用。
        #
        # 如果 compress_rate=2：
        #   当前窗口 token0-token1 的 Ca 会保存下来；
        #   处理 token2-token3 时，entry1 可以同时使用 token0-token1 的 Ca
        #   和 token2-token3 的 Cb。
        self.overlap_kv[name] = chunk_kv[:, -1, :, :head_dim].clone()
        self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
        return prior_kv, prior_gate


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear used by the grouped output projection
    The core attention's stacked output is `num_attention_heads* head_dim`-dim,
    which is *very* large (V4-Flash: 32768; V4-Pro: 65536). A direct
    `num_attention_heads*head_dim → hidden_size` projection would dominate the per-token cost.

    The paper sidesteps that by splitting the heads into `g` groups, projecting
    each `num_attention_heads * head_dim/g`-dim group independently to a `d_g`-dim intermediate output
    (with `d_g < num_attention_heads * head_dim/g`), and then mixing the resulting `g·d_g` vector to
    `hidden_size` through a single follow-up linear (`self_attn.o_b_proj`). This
    module owns the per-group block (`self_attn.o_a_proj`).

    For V4-Flash (num_attention_heads=64, head_dim=512, o_groups=8, o_lora_rank=1024,
    hidden_size=4096), g=8 groups of 4096-dim each are projected to 1024-dim, then
    mixed to 4096-dim; for V4-Pro (num_attention_heads=128, head_dim=512, o_groups=16,
    o_lora_rank=1024, hidden_size=7168), g=16 groups of 4096-dim each are projected
    to 1024-dim, then mixed to 7168-dim.
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GroupedLinear 不是普通 Linear。
        # 普通 Linear 会把所有 attention heads 拼在一起一次性投影。
        # 这里先按 group 拆开，每组用同一套 bmm 形式计算，再把组结果拼回去。
        #
        # 形状示例：
        #   x: [B,S,groups=2,hidden_per_group=4]
        #   weight reshape 后：每个 group 都有自己的 [4,out_per_group] 矩阵。
        #   输出: [B,S,groups=2,out_per_group]
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


class DeepseekV4HCACompressor(nn.Module):
    """
    Heavily Compressed Attention compressor (paper §2.3.2, eqs. 20–23). compresses
    every `compress_rate_hca` (m'=128) source tokens into a single compressed KV
    entry.

    Each closed window of m' tokens produces one compressed entry:
    `C^{Comp}_i = Σ_{j∈window} softmax(Z_j + B)_j ⊙ C_j`. RoPE on the trailing
    `rope_head_dim` slice is applied at the deterministic absolute position
    `i * compress_rate_hca + first_window_position` so cross-call concatenation
    stays causality-correct. Returns the running list of *all* compressed
    entries emitted so far (shape `[B, 1, T, head_dim]` with
    `T = entry_count["compressor"]`), so the attention can attend over the
    full long-range history.

    When `past_key_values is None` runs in stateless single-shot mode: compress
    every complete window from `hidden_states` and discard the remainder
    (instead of caching it).
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, _ = hidden_states.shape
        cache_layer: DeepseekV4HCACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
        # hidden_states 是 attention 前的普通 hidden，不是多流 mHC：
        #   [B, S, hidden_size]
        # 4-token 示例中这里的 hidden_states 为：
        #   token0=[1.274,-0.039]
        #   token1=[0.260, 0.906]
        #   token2=[0.859, 1.158]
        #   token3=[2.209, 0.993]
        #
        # kv_proj 把 hidden 映射成要被压缩的 KV 内容 C。
        # gate_proj 把 hidden 映射成每个 token 在窗口内的权重 logits Z。
        # 为了便于人工计算，示例把 kv_proj 看成恒等映射，把 gate 的 softmax 看成平均权重。
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if cache_layer is None:
            # 无 cache 时，只压缩当前 forward 中已经凑满窗口的 token。
            # compress_rate=2，S=4：
            #   usable=4
            #   chunk_kv=[token0,token1,token2,token3]
            # 如果 S=5：
            #   usable=4，第 5 个 token 不会在 stateless 模式中形成 entry。
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            # 有 cache 时，store_compression_weights 会把历史 buffer 和当前 token 拼接，
            # 只返回能整除 compress_rate 的前缀，剩余 token 留在 buffer 中。
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)

        if chunk_kv.shape[1] > 0:  # there were at least self.compress_rate tokens
            n_windows = chunk_kv.shape[1] // self.compress_rate
            # view 后按窗口组织 token：
            #   chunk_kv 原形状: [B, 4, head_dim]
            #   compress_rate=2, n_windows=2
            #   chunk_kv.view 后: [B, 2, 2, head_dim]
            #
            # 第 0 个窗口：token0-token1
            # 第 1 个窗口：token2-token3
            #
            # 示例计算：
            #   entry0 = avg(token0, token1)
            #          = [(1.274+0.260)/2, (-0.039+0.906)/2]
            #          = [0.767, 0.433]
            #   entry1 = avg(token2, token3)
            #          = [(0.859+2.209)/2, (1.158+0.993)/2]
            #          = [1.534, 1.076]
            #
            # 真实源码不是简单平均，而是下面这行 softmax 加权求和。
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias.to(
                chunk_gate.dtype
            )
            # chunk_gate.softmax(dim=2) 在窗口内做归一化。
            # dim=2 正好是窗口内部的 token 维度。
            #
            # 公式：
            #   compressed_window = sum_j softmax(gate_j + position_bias_j) * kv_j
            #
            # 若一个窗口有 token0、token1，并且 softmax 权重都是 0.5：
            #   dim0 = 0.5*1.274 + 0.5*0.260 = 0.767
            #   dim1 = 0.5*(-0.039) + 0.5*0.906 = 0.433
            #
            # kv_norm 会对压缩后的 entry 再做 RMSNorm，示例为了简化没有展开 RMSNorm。
            compressed = self.kv_norm(
                (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2)
            )
            # 每个 compressed entry 也需要位置。
            # compress_rate=2 时：
            #   entry0 代表 token0-token1，位置取 0；
            #   entry1 代表 token2-token3，位置取 2。
            # 这样 compressed KV 与 query 做 RoPE 匹配时仍然有明确的绝对位置。
            positions = torch.arange(n_windows, device=compressed.device)
            positions = (positions * self.compress_rate + first_window_position).unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            # 当前没有凑满一个窗口，就没有新的 compressed entry。
            # decode 场景中常见：compress_rate=2 时，只来一个 token，先存在 cache buffer。
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states("compressor", compressed)
        compressed_kv = compressed.unsqueeze(1)
        # compressed_kv 增加一个 KV head 维度，形状变为 [B, 1, compressed_len, head_dim]。
        # 后面 DeepseekV4Attention.forward 会执行：
        #   kv = torch.cat([sliding_kv, compressed_kv], dim=2)
        # dim=2 是 KV 序列长度维度。
        #
        # 作用：
        #   sliding_kv 保存最近的局部 token；
        #   compressed_kv 保存更长历史的摘要 entry；
        #   attention 同时读这两部分。

        compressed_len = compressed_kv.shape[2]
        seq_len = position_ids.shape[1]
        if seq_len == 1 or compressed_len == 0:
            return compressed_kv, None

        # query `t` may only see cache entries at pos `w` t > w * compress_rate (ex: t=7, w=2 t does not attend to it).
        entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
        # causal_threshold 控制每个 query 能看见多少个 compressed entry。
        # compress_rate=2，position_ids=[0,1,2,3]：
        #   causal_threshold=(position_ids+1)//2=[0,1,1,2]
        #
        # 对应含义：
        #   token0：还没有完整窗口结束，看不到 compressed entry。
        #   token1：token0-token1 已成 entry0，可以看 entry0。
        #   token2：只能看 entry0，不能看 entry1，因为 entry1 包含 token3。
        #   token3：entry0、entry1 都可以看。
        causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]
        block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
        # entry_indices >= causal_threshold 的位置写成 -inf。
        # attention softmax 遇到 -inf 后，该 KV 的概率为 0。
        #
        # compressed_len=2 时：
        #   token0 threshold=0 -> entry0/entry1 都 masked
        #   token1 threshold=1 -> entry0 可见，entry1 masked
        #   token2 threshold=1 -> entry0 可见，entry1 masked
        #   token3 threshold=2 -> entry0/entry1 都可见
        block_bias = block_bias.masked_fill(
            entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
            float("-inf"),
        )
        return compressed_kv, block_bias


class DeepseekV4Indexer(nn.Module):
    r"""Lightning Indexer (paper §2.3.1, eqs. 13–17). Used by Compressed Sparse
    Attention (CSA) to pick the top-`k` compressed KV blocks per query, with
    `k = config.index_topk`. Each query then attends only to those `k` of the
    `seq_len / compress_rate_csa` compressed entries — reduction factor
    `(seq_len / compress_rate_csa) / index_topk` over full attention against
    the entire compressed sequence.

    The indexer runs its own scaled-down compressor at `index_head_dim` over
    the same windows as the outer CSA compressor, then scores queries against
    the compressed keys with `∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)` and
    keeps the top `index_topk` indices.

    The indexer has its own rotary because it applies RoPE to two sets of
    tensors:

      * *compressed keys* at deterministic positions
        `i * compress_rate + first_window_position`,
      * *queries* at the model's current `position_ids` (variable per forward).

    Both must use the same theta as the outer compressor
    (`compress_rope_theta`) so query/key inner products are
    translation-invariant — if they used different thetas, `q · k` would carry
    a residual position-dependent skew. We can't precompute cos/sin once at
    init because the query positions vary per call, so the indexer owns its
    own rotary and calls it twice per forward (once for compressed keys, once
    for queries) with `layer_type=self.rope_layer_type` (always `"compress"`).
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.weights_scaling = self.num_heads**-0.5
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.LongTensor:
        batch, seq_len, _ = hidden_states.shape
        cache_layer: DeepseekV4CSACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
        # Indexer 是 CSA 的“检索器”。
        # 它先把历史 token 压成较小维度的 compressed_kv，
        # 再用当前 query 和这些 compressed_kv 打分，选出 top_k 个 entry。
        #
        # 最小示例：
        #   query token3 = [2.196, 0.993]
        #   entry0      = [1.009, 0.407]
        #   entry1      = [1.503, 1.077]
        #   score0 = 2.196*1.009 + 0.993*0.407 = 2.619
        #   score1 = 2.196*1.503 + 0.993*1.077 = 4.370
        #   top1  = entry1
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("indexer", kv, gate)

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            # chunk_kv: [B, S, 2*index_head_dim]
            # view 后:  [B, n_windows, ratio, 2*index_head_dim]
            #
            # 最后一维拆成两半：
            #   Ca = [..., :index_head_dim]
            #   Cb = [..., index_head_dim:]
            # CSA/indexer 都用这个双分支布局。
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

            # Same Ca / Cb overlap layout as the outer CSA compressor, at index_head_dim.
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            # 当前窗口的 Cb 放到后半段。
            # compress_rate=2 时，new_kv 的窗口长度为 4：
            #   slots 0-1：上一窗口的 Ca
            #   slots 2-3：当前窗口的 Cb
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                # 从第 1 个窗口开始，可以直接使用前一个窗口的 Ca。
                # 示例：
                #   entry1 的 slots 0-1 来自 token0-token1 的 Ca；
                #   entry1 的 slots 2-3 来自 token2-token3 的 Cb。
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state("indexer", chunk_kv, chunk_gate, self.head_dim)
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = positions * self.compress_rate + first_window_position
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        compressed_kv = (
            compressed if cache_layer is None else cache_layer.update_compressor_states("indexer", compressed)
        )

        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
        q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)

        # ReLU(q·kᵀ) * weights, then top-k
        # scores 的含义：
        #   对每个 query token，计算它和每个 compressed entry 的相关性。
        # 形状：
        #   q             [B, S, H, index_head_dim]
        #   compressed_kv [B, T, index_head_dim]
        #   scores        [B, S, H, T]
        #
        # 人工示例中省略 head 维度和 ReLU：
        #   dot([2.196,0.993], [1.503,1.077]) = 4.370
        scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling  # [B, S, H]
        # 多个 index head 的分数按 learned weights 合并成一个 entry 分数。
        # 得到 index_scores 后，topk 选出的 entry 会作为 CSA attention 能访问的长程 KV。
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]
        compressed_len = compressed_kv.shape[1]
        top_k = min(self.index_topk, compressed_len)

        # not all queries can attend to the compressed entries. If a query's position
        # is small than the relative position of the key (say m=4, query 2 cannot attend
        # to compressed key at position 4, because it compressed info for states at position
        # 12 to 16. Thus we need to make sure that top_k does not land in that range.
        # Picks that still point past `causal_threshold` (early queries with too few ready
        # blocks) are replaced with a `-1` sentinel that the compresser treats as invalid.
        if compressed_len > 0:
            # 这里的 causal_threshold 和 HCA 一样，防止 query 看到包含未来 token 的 entry。
            # compress_rate=2，position_ids=[0,1,2,3]：
            #   threshold=[0,1,1,2]
            #   token2 只能选 entry0，不能选 entry1，因为 entry1 由 token2-token3 压成，
            #   对 token2 来说包含未来 token3。
            causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]
            entry_indices = torch.arange(compressed_len, device=index_scores.device)
            future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)  # [B, S, T]
            index_scores = index_scores.masked_fill(future_mask, float("-inf"))
            # top_k_indices 是每个 query 选中的 compressed entry 下标。
            # 示例：
            #   token3 可见 entry0/entry1
            #   scores=[2.619,4.370]
            #   top1_indices=[1]
            top_k_indices = index_scores.topk(top_k, dim=-1).indices  # [B, S, k]
            invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
            return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)

        return index_scores.topk(top_k, dim=-1).indices


class DeepseekV4CSACompressor(nn.Module):
    """Compressed Sparse Attention compressor (paper §2.3.1, eqs. 9–17). Compresses
    every `compress_rate_csa` (m=4) source tokens and runs a Lightning Indexer on
    top of the compressed KV that scores queries with
    `∑_h w_{t,h} · ReLU(q_{t,h} · K^{IComp}_s)` to gather the top `index_topk`
    entries per query before they reach core attention.

    `kv_proj` / `gate_proj` / `position_bias` project to `2 * head_dim`: each
    token contributes two independent compressed series Ca and Cb stored in
    one tensor. Ca = `[..., :head_dim]` (its contribution to the *next*
    window's compressed entry), Cb = `[..., head_dim:]` (its contribution to
    the *current* window's compressed entry). Compressed entry `w` is the
    softmax-gated convex combination of window `w-1`'s Ca slice with window
    `w`'s Cb slice over `2 * compress_rate_csa` slots — width
    `2 * compress_rate_csa`, stride `compress_rate_csa`. For `w = 0` we need
    the previous window's Ca slice from the *previous forward call*; the
    cache holds it in `overlap_kv` and hands it back here. On the very first
    call (or when there is no cache) that slot stays zero-kv / `-inf`-gate,
    which gives it softmax weight 0.
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.indexer = DeepseekV4Indexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        cache_layer: DeepseekV4CSACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
        # CSA compressor 生成 attention 真正要拼接的 compressed KV。
        # 与 HCA 相比：
        #   HCA：每个窗口独立压缩，压缩率更高。
        #   CSA：使用重叠的 Ca/Cb 窗口，再由 indexer 选择少量 entry。
        #
        # 4-token 简化示例中仍可按平均值理解：
        #   entry0 = avg(token0, token1) = [1.009,0.407]
        #   entry1 = avg(token2, token3) = [1.503,1.077]
        # 真实源码使用下面的 Ca/Cb + softmax gate 生成 entry。
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            # chunk_kv 的最后一维是 2*head_dim：
            #   前 head_dim 是 Ca；
            #   后 head_dim 是 Cb。
            # view 后每个窗口有 ratio 个 token。
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

            # Lay out the two series in [B, n_win, 2*ratio, head_dim]: Cb
            # (`[..., head_dim:]`) goes in the second half (current window),
            # Ca of the previous window (`[..., :head_dim]`) goes in the
            # first half. Window 0's first half stays zero-kv / -inf-gate
            # (softmax weight 0) on the very first forward call; on later
            # calls the cache fills it with the saved Ca slice.
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            # 当前窗口的 Cb 写入后半段。
            # compress_rate=2 时：
            #   new_kv[:, window, 2:4] = 当前窗口 token 的 Cb。
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                # 前一个窗口的 Ca 写入当前窗口的前半段。
                # 这样 entry1 不只来自 token2-token3，也会收到 token0-token1 的 Ca 信息。
                # 这是 CSA 的重叠压缩。
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state(
                    "compressor", chunk_kv, chunk_gate, self.head_dim
                )
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

            # Softmax in fp32 for stability (logits in bf16/fp16 can collapse pairs that
            # only differ by a small amount, especially with large window widths).
            # dim=2 是 2*ratio 个 slot：
            #   slots 0..ratio-1     来自上一窗口 Ca；
            #   slots ratio..2*ratio 来自当前窗口 Cb。
            # softmax 后，各 slot 的权重和为 1，再对 new_kv 加权求和得到一个 entry。
            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = positions * self.compress_rate + first_window_position
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states("compressor", compressed)
        compressed_kv = compressed.unsqueeze(1)
        # compressed_kv: [B, 1, compressed_len, head_dim]。
        # 这里先不直接让 query 看所有 entry。
        # 下面的 indexer 会返回 top_k_indices，block_bias 只放行被选中的 entry。

        # Lightning Indexer: gather top-`index_topk` compressed entries per query.
        # in some cases, the output index can return top-k positions that should not be attended to.
        # Ex: for query at index 5, m=4, and `index_topk=1024`, 1024 index are return but only 2 should be
        # attended to. The indexer marks the rest with `-1`; we clamp before the gather and keep the `valid`
        # to drop them from the per-query block mask afterwards.
        top_k_indices = self.indexer(hidden_states, q_residual, position_ids, past_key_values, layer_idx)  # [B, S, k]
        compressed_len = compressed_kv.shape[2]
        valid = top_k_indices >= 0  # [B, S, k]
        # top_k_indices 中 -1 表示这个位置无效。
        # safe_indices 把 -1 替换成 compressed_len 这个额外哨兵列，避免 scatter 下标越界。
        # Per-query block bias: query `t` may only see the cache entries that are <= `seq_len // m`
        # and in these, only the ones marked valid by the indexer. Everything else is `-inf`.
        # While the above negated the indexer, here we apply the "causal" masking.
        safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
        # block_bias 初始全是 -inf，表示所有 compressed entry 都不可见。
        # scatter_ 把 indexer 选中的 entry 改成 0，表示这些位置允许 attention 使用。
        #
        # 示例：compressed_len=2，token3 的 top_k_indices=[1]
        #   初始: [-inf, -inf, -inf]  最后一列是哨兵列
        #   scatter 后: [-inf, 0, -inf]
        #   返回前去掉哨兵列: [-inf, 0]
        block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))
        block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
        return compressed_kv, block_bias[..., :compressed_len]


COMPRESSOR_CLASSES = {
    "sliding_attention": None,
    "compressed_sparse_attention": DeepseekV4CSACompressor,
    "heavily_compressed_attention": DeepseekV4HCACompressor,
}


class DeepseekV4Attention(nn.Module):
    r"""
    Diff with classic attentions:
    * Shared-KV Multi-Query Attention: `num_key_value_heads = 1`; `kv_proj` projects
      directly to that single KV head and the same tensor is read as both key and
      value.
    * Partial RoPE on the first `rope_head_dim` of each head ("Partial Rotary
      Positional Embedding"). RoPE is also applied with position `-i` to the
      attention output's rope slice, so the contribution of each KV entry stays a
      function of the *relative* distance to the query.
    * Per-head learnable attention sink like gpt OSS.
    * Grouped low-rank output projection for perfs.
    * 3 different cache mechanisms, sliding, sliding+CSA, sliding+HCA.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        # Sliding-only layers use the "main" (plain θ=10000) rope; CSA/HCA layers
        # share the same yarn-scaled "compress" rope as their compressor.
        self.rope_layer_type = "main" if self.layer_type == "sliding_attention" else "compress"
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads  # single KV head, broadcast to all
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups, config.o_groups * config.o_lora_rank, config.o_groups
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.empty(self.num_heads))
        self.compressor = (
            COMPRESSOR_CLASSES[self.layer_type](config) if self.layer_type != "sliding_attention" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # position_embeddings is a {"main", "compress"} dict from the model; pick the
        # one that matches this layer's rope type (sliding → main, CSA/HCA → compress).
        cos, sin = position_embeddings[self.rope_layer_type]

        # Attention 输入是 mHC collapse 后的一条 hidden：
        #   hidden_states: [B, S, hidden_size]
        #
        # q_a_proj/q_a_norm/q_b_proj 是 query 的低秩投影路径。
        # q_residual 会继续传给 CSA indexer，因为 indexer 也要从 query 信息打分。
        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        # DeepSeekV4 使用 shared-KV：只有 1 个 KV head。
        # kv 同时作为 key 和 value 使用，所以下面 attention_interface 传入的是 kv, kv。
        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        if past_key_values is not None:  # sliding where K==V
            # 更新滑动窗口 KV cache。
            # 这部分保存最近 sliding_window 范围内的 token。
            # 例如窗口大小为 2，query 是 token2：
            #   local window = token1-token2
            #   若 token1=[0.260,0.906]，token2=[0.859,1.158]
            #   简化平均 local=[(0.260+0.859)/2, (0.906+1.158)/2]
            #                =[0.559,1.032]
            # 真实 attention 不是平均，而是 QK^T -> softmax -> 加权求和 V。
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        block_bias = None
        if self.compressor is not None:  # Compressed KV (CSA or HCA)
            # HCA/CSA 层除了局部 sliding KV，还会生成长程 compressed KV。
            #
            # HCA 示例：
            #   token0-token1 -> entry0=[0.767,0.433]
            #   token2-token3 -> entry1=[1.534,1.076]
            #
            # 对 token2 来说：
            #   local window 是 token1-token2，提供最近上下文；
            #   visible compressed entry 是 entry0，提供更早历史摘要；
            #   entry1 还不可见，因为 entry1 包含 token3。
            compressed_kv, block_bias = self.compressor(
                hidden_states, q_residual, position_ids, past_key_values, self.layer_idx
            )
            # 在 KV 序列长度维度拼接：
            #   kv:            [B, 1, local_len, head_dim]
            #   compressed_kv: [B, 1, compressed_len, head_dim]
            #   cat 后:        [B, 1, local_len + compressed_len, head_dim]
            #
            # 拼接后，attention 会同时从局部 token 和 compressed entries 中取信息。
            kv = torch.cat([kv, compressed_kv], dim=2)

        # The compressor path concatenates extra entries onto the KV axis after the
        # standard sliding-window cache update, so a tensor `attention_mask` (built
        # for the pre-concat KV length) needs to be extended to cover them. The
        # compressor returns a `block_bias` carrying per-query causality + indexer
        # validity over those new slots — cat it in instead of zero-padding (which
        # would let every query see every compressed slot).
        if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
            if block_bias is not None:
                attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
            else:
                attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        # attention_interface 执行真正的注意力计算：
        #   scores = q @ k^T * scaling + attention_mask
        #   probs  = softmax(scores)
        #   output = probs @ v
        #
        # 对 HCA/CSA 层，k/v 已经包含两部分：
        #   1. sliding local KV；
        #   2. compressed KV。
        # block_bias 已经被拼到 attention_mask 中，用来禁止访问未来 compressed entry，
        # 或禁止访问 CSA indexer 没选中的 entry。
        attn_output, attn_weights = attention_interface(
            self,
            q,
            kv,
            kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        # K=V in V4, so V picked up rope on its trailing rope slice. Apply the conjugate
        # rotation (`-sin`) at the query position to undo it on the rope slice of the
        # output before the grouped output projection mixes heads. The transpose pair is
        # just a layout fix-up: apply_rotary_pos_emb expects `[B, S, H, D]` (its
        # `unsqueeze_dim=1` adds a head-broadcast dim to cos/sin); attention gave us
        # `[B, H, S, D]`.
        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)

        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, attn_weights


class DeepseekV4HyperConnection(nn.Module):
    r"""
    Manifold-Constrained Hyper-Connections
    (mHC) (Xie et al., 2026) to strengthen the conventional residual connections between adjacent
    Transformer blocks

    Owns the learned (`fn`, `base`, `scale`)
    parameters that turn the incoming `hc_mult` residual streams into collapse / expand
    weights. The decoder layer instantiates two of these (one for the attention site,
    one for the mlp site).

    ASCII shape guide — `B` = batch, `S` = seq, `H` = hc_mult, `D` = hidden_size::

              hidden_streams        flatten(2)        RMSNorm-rescale + F.linear(fn)
         [B, S, H, D]  ──────────►  [B, S, H*D]  ─────────────────────────────────►
                                                             mix-logits
                                                             [B, S, (2+H)*H]
                                                                    │
                            ┌───────────────────────────────────────┴──────────────────────────────┐
                            ▼                          ▼                                           ▼
                        pre logits                post logits                               comb logits
                        [B, S, H]                 [B, S, H]                                 [B, S, H, H]
                        × scale[0]                × scale[1]                                × scale[2]
                        + base[:H]                + base[H:2H]                              + base[2H:]
                        σ() + eps                 σ() + eps                                 σ() + eps
                        │                         │                                         │
                        pre                        post                                     Sinkhorn(iters)
                        (stream collapse weights)  (block-output placement)                 row/col normalise
                                                                                            │
                                                                                            comb
                                                                                            (stream mixer)
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # 3 = number of outputs from the mHC mapping: `pre` (input projection
        # weights), `post` (sublayer output projection weights), `comb` (the
        # H×H residual combine matrix that gets Sinkhorn-projected onto the
        # doubly-stochastic manifold). Each output gets its own learned scale.
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Compute `pre`, `post`, `comb` from the mHC mapping (paper §2.2 eq. 8).
        `comb` is projected onto the doubly-stochastic manifold via Sinkhorn-
        Knopp: starting from the sigmoid-positive matrix, alternate row and
        column normalisation for `hc_sinkhorn_iters` steps. `pre` then collapses
        the `hc_mult` parallel streams into a single sequence (input projection
        into the sublayer); `post` and `comb` are returned for the caller to
        apply on the sublayer output.
        """
        hc = self.hc_mult
        # hidden_streams 的形状始终是 [B, S, hc_mult, hidden_size]。
        # hc_mult=2 时，一个 token 有两条残差流：
        #   stream0=[1.000, 0.000]
        #   stream1=[1.100,-0.100]
        #
        # flatten(start_dim=2) 把两条流拼到最后一维：
        #   [stream0, stream1] -> [1.000,0.000,1.100,-0.100]
        # 这样 fn 可以根据所有流的信息，计算 pre/post/comb 三组混合参数。
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        # pre：把 hc_mult 条流折叠成一条 hidden，送入 attention 或 MLP。
        # 示例中为了好算，run_deepseek_v4_tiny_trace.py 用平均权重：
        #   pre=[0.5,0.5]
        #   collapsed = 0.5*stream0 + 0.5*stream1
        #             = [1.05,-0.05]
        # 真实源码中 pre 由线性层 + sigmoid 学出来。
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        # post：控制子层输出写回每条残差流的比例。
        # 如果某个 token 的 attention 输出为 [1.05,-0.05]，
        # post=[0.3,0.6] 表示：
        #   stream0 接收 0.3 * attention_output；
        #   stream1 接收 0.6 * attention_output。
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        # comb：控制旧残差流之间如何互相混合。
        # hc_mult=2 时，comb 是 2x2 矩阵。
        # 如果简化成对角保持：
        #   comb = [[0.7,0.0],
        #           [0.0,0.4]]
        # 再加 post 写入，就得到 trace 中的简化规则：
        #   stream0_new = 0.7*stream0_old + 0.3*sublayer_output
        #   stream1_new = 0.4*stream1_old + 0.6*sublayer_output
        #
        # 真实源码用 Sinkhorn 归一化 comb，让每行、每列的总量受控制。
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        # Collapse the `hc_mult` parallel streams down to a single sequence using
        # the `pre` weights: one weighted sum across the stream axis, ready for
        # the sublayer (attn / MLP).
        # collapsed 的形状从 [B,S,hc_mult,D] 变为 [B,S,D]。
        # 后面的 attention 和 MLP 都只接收这一条 hidden。
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


class DeepseekV4HyperHead(nn.Module):
    """Final HC-stream collapse; used by `DeepseekV4Model` before the shared RMSNorm."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.eps = config.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 最后一层之后，模型还保留 hc_mult 条残差流。
        # HyperHead 的作用是把多流折叠回普通 hidden：
        #   [B,S,hc_mult,D] -> [B,S,D]
        # 然后 DeepseekV4Model.forward 再接 self.norm。
        #
        # 简化示例：
        #   token0 stream0=[1.700,-0.034]
        #   token0 stream1=[2.003,-0.037]
        #   平均折叠 -> [(1.700+2.003)/2, (-0.034-0.037)/2]
        #             = [1.851,-0.035]
        flat = self.input_norm(x.flatten(2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


class DeepseekV4MLP(LlamaMLP):
    pass


@use_experts_implementation
class DeepseekV4Experts(MixtralExperts):
    # GPT OSS style, no bias

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.limit = config.swiglu_limit

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # Lives on the class (like gpt-oss's _apply_gate) so the grouped_mm / batched_mm
        # backends swapped in by `@use_experts_implementation` apply the same clamp +
        # SiLU on top of their packed gate_up output instead of bypassing it.
        gate, up = gate_up.chunk(2, dim=-1)
        # Expert 内部是 SwiGLU：
        #   gate_up_proj 一次性算出 gate 和 up 两段；
        #   gate 先经过 SiLU 激活；
        #   再与 up 逐元素相乘。
        #
        # 简化示例没有展开 SwiGLU，而是把 expert0 写成：
        #   expert0([x,y]) = [1.4*x, 0.7*y]
        # 真实源码中 expert 输出由 gate_up_proj、SiLU、down_proj 三步得到。
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        # hidden_states 已经被展平成 [B*S, hidden_dim]。
        # top_k_index:   每个 token 选中的专家编号，例如 token0 选 [0,2]。
        # top_k_weights: 每个专家对应的权重，例如 [0.676,0.324]。
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            # mask[e, k, t] 表示第 t 个 token 的第 k 个路由位置是否选择 expert e。
            # 这样可以按 expert 分组，把同一个 expert 要处理的 token 一次取出来。
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            # current 是该 expert 对这些 token 的输出，然后乘对应 router 权重。
            #
            # 最小示例 token0：
            #   input=[1.042,-0.043]
            #   selected expert0 weight=0.676
            #   selected expert2 weight=0.324
            #
            #   expert0 output=[1.4*1.042, 0.7*(-0.043)]
            #                 =[1.459,-0.030]
            #   expert2 output=[1.1*1.042, 1.1*(-0.043)]
            #                 =[1.147,-0.047]
            #
            #   routed dim0 = 0.676*1.459 + 0.324*1.147 = 1.358
            #   routed dim1 = 0.676*(-0.030) + 0.324*(-0.047) = -0.035
            current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
            # 同一个 token 会被多个 expert 处理。
            # index_add_ 把多个 expert 的加权输出加回同一个 token 位置。
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


class DeepseekV4TopKRouter(MixtralTopKRouter):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        # learned gate：
        #   logits = hidden @ router_weight.T
        #   scores = score_fn(logits)
        #
        # 简化示例中把 scores 写成：
        #   scores=[x, y, 0.5*(x+y)]
        # 对 input=[1.042,-0.043]：
        #   scores=[1.042,-0.043,0.500]
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        # topk 按分数选择专家。
        # 示例：
        #   scores=[1.042,-0.043,0.500]
        #   top2 -> expert0 和 expert2
        indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        # 只在选中的专家之间归一化。
        # 示例：
        #   selected_scores=[1.042,0.500]
        #   sum=1.542
        #   weights=[1.042/1.542, 0.500/1.542]
        #          =[0.676,0.324]
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4HashRouter(MixtralTopKRouter):
    r"""
    Hash routing for the first `mlp_layer_types == "hash_moe"` MoE layers (paper
    §2.1). Expert selection is determined by a fixed `tid2eid[input_ids]` lookup —
    a frozen token-id → expert-id table — instead of a learned argmax. The learned
    gate `weight` still produces the per-expert scores that weight the selected
    experts' activations; only the *which-experts* selection is static.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer("tid2eid", torch.zeros(config.vocab_size, self.top_k, dtype=torch.long), persistent=True)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        # hash_moe 不用 learned scores 决定“选谁”。
        # 它直接用 input_ids 查 tid2eid 表：
        #   token_id -> 固定的 top_k expert ids
        # scores 仍然用于给这些固定 expert 计算权重。
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4SparseMoeBlock(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
        self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = DeepseekV4MLP(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        # MoE 输入是一条普通 hidden：
        #   [B,S,D]
        # 先展平成 [B*S,D]，每个 token 独立选择专家。
        if self.is_hash:
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        # routed 是被选中的稀疏专家输出。
        #
        # token0 简化例子：
        #   input=[1.042,-0.043]
        #   router 选择 expert0/expert2，weights=[0.676,0.324]
        #   routed=[1.358,-0.035]
        routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
        # shared_experts 是每个 token 都会经过的共享 MLP。
        # 简化示例中 shared=0.2*input：
        #   shared=[0.209,-0.009]
        #   MoE output = routed + shared
        #              = [1.358,-0.035] + [0.209,-0.009]
        #              = [1.567,-0.044]
        return routed + self.shared_experts(residual)


class DeepseekV4DecoderLayer(GradientCheckpointingLayer):
    r"""DeepSeek-V4 decoder block (paper §2). Differs from a classic residual block in
    two places:

    The residual is a stack of `hc_mult` parallel streams kept in shape
    `[B, S, hc_mult, D]` throughout the block, mixed in and out via two
    :class:`DeepseekV4HyperConnection` modules (Manifold-Constrained Hyper-
    Connections / mHC, paper §2.2; Xie et al., 2026). The mHC mappings constrain
    the residual transform to the manifold of doubly-stochastic matrices via the
    Sinkhorn-Knopp projection — making signal propagation non-expansive across
    deep stacks.

    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV4Attention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # hidden_states throughout: [B, S, hc_mult, hidden].
        #
        # 一个 DecoderLayer 的执行顺序固定如下：
        #   1. attn_hc 把多条残差流折叠成 collapsed，送入 attention。
        #   2. self_attn 计算 attention 输出。
        #   3. post + comb 把 attention 输出混回多条残差流。
        #   4. ffn_hc 再次把多条残差流折叠成 collapsed，送入 MoE。
        #   5. mlp 计算 MoE 输出。
        #   6. post + comb 把 MoE 输出混回多条残差流。
        #
        # 形状变化：
        #   hidden_states: [B,S,hc_mult,D]
        #   collapsed:     [B,S,D]
        #   attn_output:   [B,S,D]
        #   mlp_output:    [B,S,D]
        #   return:        [B,S,hc_mult,D]
        #
        # `post` / `comb` come out of the HC modules in fp32 (Sinkhorn projection runs
        # in float); the .to(dtype) puts everything back to the input dtype before mixing
        # so both sites stay consistent with `hidden_states`'s entry dtype.
        # comb is consumed transposed: indexed as sum_j comb[j, k] * residual[j, d]
        # (sum over the FIRST hc axis), equivalent to comb.T @ residual. Sinkhorn
        # produces a doubly-stochastic but non-symmetric matrix, so the direction matters.
        dtype = hidden_states.dtype
        # 第一次 mHC：attention 前折叠。
        # 示例 token0：
        #   stream0=[1.000,0.000]
        #   stream1=[1.100,-0.100]
        #   collapsed=[(1.000+1.100)/2, (0.000-0.100)/2]
        #            =[1.050,-0.050]
        post, comb, collapsed = self.attn_hc(hidden_states)
        # input_layernorm 先归一化 collapsed，再送入 self_attn。
        # self_attn 根据当前层类型执行：
        #   sliding_attention：只使用局部窗口；
        #   heavily_compressed_attention：局部窗口 + HCA compressed KV；
        #   compressed_sparse_attention：局部窗口 + CSA compressed KV + indexer。
        attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
        # 把 attention 输出写回多条残差流。
        # 简化例子：
        #   stream0_new = 0.7*stream0_old + 0.3*attn_output
        #   stream1_new = 0.4*stream1_old + 0.6*attn_output
        #
        # 源码中：
        #   post * attn_output       表示“写入子层输出”；
        #   comb.T @ hidden_states   表示“旧残差流之间重新混合”。
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

        # 第二次 mHC：MoE 前折叠。
        # 注意这里使用的是 attention 更新后的 hidden_states。
        post, comb, collapsed = self.ffn_hc(hidden_states)
        # post_attention_layernorm 归一化 collapsed 后送入 MoE。
        # MoE 会执行 router 选专家、experts 加权求和、shared_experts 相加。
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        # 把 MoE 输出写回多条残差流，返回给下一层。
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )


class DeepseekV4PreTrainedModel(MixtralPreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV4DecoderLayer"]
    # V4 ships eager-only. The non-eager backends are off for the following reasons:
    #
    #   * FlashAttention 2 / 3 cap the head dim at 256; V4's `head_dim=512`
    #     (V4-Flash and V4-Pro both) is structurally incompatible — `flash_attention_2`
    #     and the `kernels-community/vllm-flash-attn3` kernel both fail with
    #     `RuntimeError: FlashAttention forward only supports head dimension at most
    #     256`. FA4 has the same 256 cap, so it's off too.
    #   * SDPA: torch's SDPA kernel doesn't carry the per-head learnable sink term V4
    #     inherits from gpt-oss-style attention.
    #   * FlexAttention: V4 attention concatenates compressor entries onto the KV
    #     axis *inside* the attention block, after the model-level mask was built,
    #     so the resulting KV length doesn't match the BlockMask's `kv_len`.
    #     BlockMask has no runtime resize, and rebuilding it per-block would require
    #     teaching the compressor's variable output count to a `mask_mod` — not
    #     worth it for a path the compressor already owns its own causality
    #     bookkeeping for.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    # The compressor's rolling-window buffer / compressed-entries / overlap state
    # lives on the per-layer cache (:class:`DeepseekV4HCACache` /
    # :class:`DeepseekV4CSACache`) and isn't compatible with :class:`StaticCache`
    # — that path would hand the compressor a :class:`StaticSlidingWindowLayer`
    # with no `store_compression_weights` method. Disabling fullgraph compile
    # keeps generation tests on the dynamic cache build that does dispatch to
    # V4's own cache layers.
    _can_compile_fullgraph = False
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "e_score_correction_bias",
        "q_a_norm",
        "kv_norm",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    ]
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
    # ``_is_stateful`` opts out of generation modes that need to roll the cache
    # back across drafts (assisted generation, prompt lookup, contrastive search).
    # The compressor's running-window state isn't rewindable, so `generate`
    # raises a clear error early instead of failing deep in the compressor with
    # a missing-method `AttributeError`.
    _is_stateful = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(DeepseekV4TopKRouter, index=0),
        "hidden_states": DeepseekV4DecoderLayer,
        "attentions": DeepseekV4Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, (DeepseekV4TopKRouter, DeepseekV4HashRouter)):
            init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, DeepseekV4TopKRouter):
                init.zeros_(module.e_score_correction_bias)  # buffer
            if isinstance(module, DeepseekV4HashRouter):
                init.zeros_(module.tid2eid)  # buffer; real values come from the checkpoint
        elif isinstance(module, DeepseekV4Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Attention):
            init.zeros_(module.sinks)
        elif isinstance(module, DeepseekV4HyperConnection):
            init.normal_(module.fn, mean=0.0, std=std)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, DeepseekV4HyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        elif isinstance(module, (DeepseekV4HCACompressor, DeepseekV4CSACompressor, DeepseekV4Indexer)):
            init.zeros_(module.position_bias)
        elif isinstance(module, DeepseekV4RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


@auto_docstring
class DeepseekV4Model(LlamaModel):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [DeepseekV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.hc_head = DeepseekV4HyperHead(config)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        # Model.forward 是 DeepSeekV4 主干入口。
        # 典型调用：
        #   input_ids = [[1,2,3,4]]
        # 执行路径：
        #   token id -> embedding -> mHC 多流 -> N 个 DecoderLayer
        #            -> HyperHead 折叠 -> RMSNorm -> last_hidden_state
        #
        # lm_head 不在这个类中，语言模型 logits 由 DeepseekV4ForCausalLM 负责。
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        return_cache = past_key_values if use_cache else None
        if past_key_values is None:
            # DynamicCache 会根据 config.layer_types 给每层创建不同 cache：
            #   sliding_attention              -> 普通滑窗 cache
            #   heavily_compressed_attention   -> DeepseekV4HCACache
            #   compressed_sparse_attention    -> DeepseekV4CSACache
            # 压缩 attention 的 buffer、compressed_kv、overlap 都存在这里。
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            # token id 查 embedding 表。
            # 示例：
            #   input_ids=[1,2,3,4]
            #   embedding_table[1]=[1,0]
            #   embedding_table[2]=[0,1]
            #   inputs_embeds=[[1,0],[0,1],...]
            # 真实 embedding 维度是 hidden_size，不是示例中的 2。
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            # position_ids 是每个 token 的绝对位置。
            # prefill 时通常是 [0,1,2,3]。
            # decode 时 past_seen 不为 0，新 token 会接着历史位置编号。
            past_seen = past_key_values.get_seq_length()
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)
            # `generate()` may pass a per-layer-type mask dict already built by
            # `create_masks_for_generate`; all V4 layer types use the same sliding-window
            # mask, so use the prebuilt one directly. Otherwise build it here.
        if isinstance(attention_mask, dict):
            causal_mask = next(iter(attention_mask.values()))
        else:
            # causal_mask 是普通滑动窗口 attention 的 mask。
            # HCA/CSA 额外拼接 compressed KV 后，会在 DeepseekV4Attention.forward
            # 中把 compressor 返回的 block_bias 再拼到这个 mask 后面。
            causal_mask = create_sliding_window_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        # mHC 要求每个 token 有 hc_mult 条残差流。
        # inputs_embeds:  [B,S,D]
        # unsqueeze(2):   [B,S,1,D]
        # expand 后:      [B,S,hc_mult,D]
        #
        # 示例 hc_mult=2：
        #   token0 hidden=[1,0]
        #   扩成两条初始流：
        #     stream0=[1,0]
        #     stream1=[1,0]
        # run_deepseek_v4_tiny_trace.py 为了观察差异，把 stream1 人为加了偏移；
        # 真实源码这里两条初始流相同，后续由每层 mHC 混合出差异。
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        # 预先计算两套 RoPE：
        #   main：普通 sliding attention 使用；
        #   compress：HCA/CSA compressor、indexer、compressed attention 使用。
        # 每层在 DeepseekV4Attention.forward 中按 self.rope_layer_type 取对应 RoPE。
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }

        for layer in self.layers:
            # 每一层都接收并返回 [B,S,hc_mult,D]。
            # 层内才会临时折叠成 [B,S,D] 去做 attention 和 MoE。
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask,
                input_ids=input_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        # 所有 DecoderLayer 执行完后，仍然是多条残差流。
        # hc_head 把 [B,S,hc_mult,D] 折叠成 [B,S,D]，
        # norm 再做最后一次 RMSNorm，输出给上层 CausalLM 的 lm_head。
        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=return_cache)


class DeepseekV4ForCausalLM(MixtralForCausalLM):
    pass


__all__ = [
    "DeepseekV4PreTrainedModel",
    "DeepseekV4Model",
    "DeepseekV4ForCausalLM",
]
