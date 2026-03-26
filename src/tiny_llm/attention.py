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
        pass

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        pass


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
