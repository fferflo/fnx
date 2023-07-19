import fnx, jax, einx
import jax.numpy as jnp

def split_heads(x, heads):
    if x.shape[-1] % heads != 0:
        raise ValueError(f"Channel dimension {x.shape[-1]} must be divisible by number of heads {heads}")
    return einx.rearrange("b... t (h c) -> b... h t c", x, h=heads)

def merge_heads(x):
    return einx.rearrange("b... h t c -> b... t (h c)", x)

def multihead_attention_function(func):
    def outer(query, key, value, heads=1, **kwargs):
        # Split heads
        query = split_heads(query, heads)
        key = split_heads(key, heads)
        value = split_heads(value, heads)

        x = func(query, key, value, **kwargs)

        # Combine heads
        x = merge_heads(x)

        return x
    outer.__name__ = func.__name__
    outer = fnx.module(outer)
    return outer

# More attention types: https://github.com/idiap/fast-transformers/tree/master/fast_transformers/attention

# https://arxiv.org/abs/1706.03762
@multihead_attention_function
def full_attention(query, key, value, prior_logits=None, mask=None): # TODO: add q_axis, k_axis etc arguments?
    query = query * (query.shape[-1] ** -0.5)
    weights = einx.dot("b... tq c, b... tk c -> b... tq tk", query, key)

    if not mask is None:
        weights = jnp.where(mask, weights, -jnp.inf)
    if not prior_logits is None:
        weights = weights + prior_logits

    fnx.sow("logits", weights)
    weights = jax.nn.softmax(weights, axis=-1)
    fnx.sow("weights", weights)
    return einx.dot("b... tq tk, b... tk c -> b... tq c", weights, value)

class mask:
    def causal(tokens_num):
        return jnp.tril(jnp.ones((tokens_num, tokens_num), dtype=bool))

    def local_causal(tokens_num, window_size): # Only use the last window_size tokens
        result = mask.causal(tokens_num)
        result = result != jnp.tril(result, -window_size)
        return result
