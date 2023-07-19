import fnx, math, einx
import jax.numpy as jnp
import numpy as np
import haiku as hk

@fnx.module
def positional_embed(x, temperature=10000.0, epsilon=1e-6):
    num_pos_feats = x.shape[-1] // (len(x.shape) - 2)
    if num_pos_feats * (len(x.shape) - 2) != x.shape[-1]:
        raise ValueError("Channels must be divisible by number of spatial dimensions")

    coords = jnp.stack(jnp.meshgrid(
        *[jnp.arange(s).astype("int32") for s in x.shape[1:-1]],
        indexing="ij",
    ), axis=-1) # s... i
    coords = einx.rearrange("s... i -> (s...) i", coords)
    coords = coords.astype("float32") * (2 * math.pi / (np.asarray(x.shape[1:-1]) - 1 + epsilon))[np.newaxis, :]

    y = temperature ** (2 * (jnp.arange(num_pos_feats) // 2) / num_pos_feats)
    y = coords[:, :, jnp.newaxis] / y[jnp.newaxis, jnp.newaxis, :] # (s...) i f

    y = jnp.stack([
        jnp.sin(y[..., 0::2]),
        jnp.cos(y[..., 1::2]),
    ], axis=-1) # (s...) i f//2 2

    y = einx.rearrange("(s...) c... -> s... (c...)", y, s=x.shape[1:-1])
    assert y.shape[-1] == x.shape[-1]

    return x + y[jnp.newaxis]

@fnx.module
def self_attention(x0, qk, v, heads):
    qk = einx.rearrange("b s... c -> b (s...) c", qk)
    v = einx.rearrange("b s... c -> b (s...) c", v)

    qk = fnx.linear(qk, channels=2 * x0.shape[-1], bias=True, name="qk")
    q, k = jnp.split(qk, indices_or_sections=2, axis=-1)
    v = fnx.linear(v, channels=1 * x0.shape[-1], bias=True, name="v")
    x = fnx.attention.full_attention(q, k, v, heads=heads)
    x = fnx.linear(x, channels=x0.shape[-1], bias=True, name="out")

    x = einx.rearrange("b (s...) c -> b s... c", x, s=x0.shape[1:-1])

    x = x0 + x
    x = fnx.norm(x)

    return x

@fnx.module
def cross_attention(x0, q, k, v, heads):
    q = einx.rearrange("b s... c -> b (s...) c", q)
    k = einx.rearrange("b s... c -> b (s...) c", k)
    v = einx.rearrange("b s... c -> b (s...) c", v)

    q = fnx.linear(q, channels=1 * x0.shape[-1], bias=True, name="q")
    k = fnx.linear(k, channels=1 * x0.shape[-1], bias=True, name="k")
    v = fnx.linear(v, channels=1 * x0.shape[-1], bias=True, name="v")
    x = fnx.attention.full_attention(q, k, v, heads=heads)
    x = fnx.linear(x, channels=x0.shape[-1], bias=True, name="out")

    x = einx.rearrange("b (s...) c -> b s... c", x, s=x0.shape[1:-1])

    x = x0 + x
    x = fnx.norm(x)

    return x

@fnx.module
def mlp(x, mlp_ratio=4):
    x0 = x

    x = fnx.linear(x, channels=int(x0.shape[-1] * mlp_ratio), bias=True, name="linear1")
    x = fnx.act(x)
    x = fnx.linear(x, channels=x0.shape[-1], bias=True, name="linear2")

    x = x0 + x
    x = fnx.norm(x)

    return x

@fnx.module
def encode_block(x, heads):
    x = self_attention(
        x0=x,
        qk=positional_embed(x, temperature=10000.0, epsilon=1e-6),
        v=x,
        heads=heads,
    )
    x = mlp(x)
    return x

@fnx.module
def decode_block(x, v, x_embed, heads):
    x = self_attention(
        x0=x,
        qk=x + x_embed[jnp.newaxis],
        v=x,
        heads=heads,
    )
    x = cross_attention(
        x0=x,
        q=x + x_embed[jnp.newaxis],
        k=positional_embed(v, temperature=10000.0, epsilon=1e-6),
        v=v,
        heads=heads,
    )
    x = mlp(x)
    return x

@fnx.module
def encode(x, depth, heads):
    for block_index in range(depth):
        x = encode_block(x, heads=heads, name=f"block{block_index + 1}")
    return x

@fnx.module
def decode(v, num_queries, depth, heads):
    x_embed = fnx.param("query_embed", shape=(num_queries, v.shape[-1]), dtype=v.dtype, init=hk.initializers.TruncatedNormal())
    x = jnp.zeros((v.shape[0], x_embed.shape[0], x_embed.shape[1]))

    for block_index in range(depth):
        x = decode_block(
            x=x,
            v=v,
            x_embed=x_embed,
            heads=heads,
            name=f"block{block_index + 1}",
        )

    return x
