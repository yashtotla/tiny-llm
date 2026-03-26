import mlx.core as mx
from .basics import softmax, linear


# Attn(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
# O(L·S) memory on the score matrix — the bottleneck Flash Attention + KV-caches address.
def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    scale = scale or mx.rsqrt(query.shape[-1])
    # MLX gotcha: .transpose() reverses ALL axes; swapaxes only swaps the last two.
    scores = query @ key.swapaxes(-2, -1) * scale
    if mask is not None:
        scores = scores + mask
    return softmax(scores, axis=-1) @ value


# MHA: project → split heads → per-head attention → merge heads → project out.
# Shape journey: (N, L, E) → (N, L, H, D) → (N, H, L, D) → attn → back.
# In real inference, the KV projections get cached (KV-cache) so we only
# compute them once per token during autoregressive decoding.
class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape
        query_proj = linear(query, self.wq)
        key_proj = linear(key, self.wk)
        value_proj = linear(value, self.wv)

        # (N, L, E) → (N, H, L, D): each head attends independently over its subspace.
        q = query_proj.reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = key_proj.reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = value_proj.reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        output = scaled_dot_product_attention_simple(q, k, v, mask=mask)
        # (N, H, L, D) → (N, L, E): merge heads back before output projection.
        output = output.transpose(0, 2, 1, 3).reshape(N, L, self.hidden_size)
        return linear(output, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
