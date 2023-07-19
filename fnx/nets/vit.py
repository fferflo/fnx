import fnx, jax, einx
import jax.numpy as jnp
import numpy as np
from functools import partial
import haiku as hk

@fnx.module
def patch_embed(x, channels, patch_size):
    x = fnx.conv(x, channels=channels, kernel_size=patch_size, stride=patch_size, bias=True, padding=0) # TODO: bias should be false
    x = fnx.norm(x)
    return x

@fnx.module
def positional_embed(x, shape, resize_method="bicubic"):
    if shape is None:
        shape = x.shape[1:-1]
    param = fnx.param("positional_embed", shape=tuple(shape) + (x.shape[-1],), dtype=x.dtype, init=hk.initializers.RandomNormal(stddev=2e-2))
    param = jax.image.resize(param, x.shape[1:], method=resize_method, antialias=False)
    return x + param[jnp.newaxis]

@fnx.module
def add_prefix_tokens(x, num):
    x = einx.rearrange("b s... c -> b (s...) c", x)
    if num > 0:
        prefix_tokens = fnx.param("prefix_tokens", shape=[num, x.shape[-1]], dtype=x.dtype, init=hk.initializers.RandomNormal(stddev=2e-2))
        x = jnp.concatenate([prefix_tokens[jnp.newaxis], x], axis=1)
    return x

@fnx.module
def block(x, mlp_ratio=4, heads=1):
    with fnx.scope("attn"):
        x0 = x
        x = fnx.norm(x)

        x = fnx.linear(x, channels=3 * x0.shape[-1], bias=True, name="qkv")
        query, key, value = jnp.split(x, indices_or_sections=3, axis=-1)
        x = fnx.attention.full_attention(query, key, value, heads=heads)
        x = fnx.linear(x, channels=x0.shape[-1], bias=True, name="out")

        x = fnx.layerscale(x)
        x = x0 + fnx.stochastic.droppath(x)

    with fnx.scope("mlp"):
        x0 = x
        x = fnx.norm(x)

        x = fnx.linear(x, channels=int(x0.shape[-1] * mlp_ratio), bias=True, name="linear1")
        x = fnx.act(x)
        x = fnx.linear(x, channels=x0.shape[-1], bias=True, name="linear2")

        x = fnx.layerscale(x)
        x = x0 + fnx.stochastic.droppath(x)

    return x

@fnx.module
def encode(x, depth, channels, patch_size=16, heads=1, pos_embed_shape=None, prefix_tokens=0, block=block):
    x = patch_embed(x, channels=channels, patch_size=patch_size)
    x = positional_embed(x, shape=pos_embed_shape)
    x = add_prefix_tokens(x, prefix_tokens)

    for block_index in range(depth):
        x = block(x, heads=heads, name=f"block{block_index + 1}")
    x = fnx.norm(x)

    return x



def vit_tiny(x, **kwargs):
    return encode(
        x,
        depth=12,
        channels=192,
        heads=3,
        **kwargs,
    )

def vit_small(x, **kwargs):
    return encode(
        x,
        depth=12,
        channels=384,
        heads=6,
        **kwargs,
    )

def vit_medium(x, **kwargs):
    return encode(
        x,
        depth=12,
        channels=512,
        heads=8,
        **kwargs,
    )

def vit_base(x, **kwargs):
    return encode(
        x,
        depth=12,
        channels=768,
        heads=12,
        **kwargs,
    )

def vit_large(x, **kwargs):
    return encode(
        x,
        depth=24,
        channels=1024,
        heads=16,
        **kwargs,
    )

def vit_huge(x, **kwargs):
    return encode(
        x,
        depth=32,
        channels=1280,
        heads=16,
        **kwargs,
    )

def vit_giant(x, **kwargs):
    return encode(
        x,
        depth=40,
        channels=1408,
        heads=16,
        block=partial(block, mlp_ratio=48 / 11),
        **kwargs,
    )

def vit_giant2(x, **kwargs):
    return encode(
        x,
        depth=40,
        channels=1536,
        heads=24,
        block=partial(block, mlp_ratio=8 / 3),
        **kwargs,
    )

def vit_gigantic(x, **kwargs):
    return encode(
        x,
        depth=48,
        channels=1664,
        heads=16,
        block=partial(block, mlp_ratio=64 / 13),
        **kwargs,
    )
