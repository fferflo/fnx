import fnx
import jax.numpy as jnp
from functools import partial
import haiku as hk

@fnx.module
def vocabulary_embed(x, channels, size, init=hk.initializers.TruncatedNormal()):
    vocab_embed = fnx.param("weight", shape=(size, channels), dtype="float32", init=init)
    return vocab_embed[x]

@fnx.module
def positional_embed(x, size, init=hk.initializers.Constant(0.0)):
    positional_embed = fnx.param("weight", shape=(size, x.shape[-1]), dtype=x.dtype, init=init)
    if x.shape[-2] > size:
        raise ValueError(f"Number of tokens {x.shape[-2]} is larger than block size {size}")

    return x + positional_embed[jnp.newaxis, :x.shape[-2], :]

@fnx.module
def block(x, channels=None, mlp_ratio=4, heads=1):
    if channels is None:
        channels = x.shape[-1]

    # Self-attention
    with fnx.scope("attn"):
        x0 = x
        x = fnx.norm(x)

        x = fnx.linear(x, channels=3 * channels, bias=True, name="qkv")
        query, key, value = jnp.split(x, indices_or_sections=3, axis=-1)
        x = fnx.attention.full_attention(query, key, value, heads=heads, mask=fnx.attention.mask.causal(x.shape[-2]))
        x = fnx.linear(x, channels=channels, bias=True, name="out")

        x = x0 + fnx.stochastic.droppath(x)

    # MLP
    with fnx.scope("mlp"):
        x0 = x
        x = fnx.norm(x)

        x = fnx.linear(x, channels=int(channels * mlp_ratio), bias=True, name="linear1")
        x = fnx.act(x)
        x = fnx.linear(x, channels=channels, bias=True, name="linear2")

        x = x0 + fnx.stochastic.droppath(x)

    return x

@fnx.module
def encode(x, depth, channels, vocab_size, block_size, heads=1, block=block): # TODO: move heads into block?
    x = vocabulary_embed(x, size=vocab_size, channels=channels)
    x = positional_embed(x, size=block_size)

    for block_index in range(depth):
        x = block(
            x,
            heads=heads,
            name=f"block{block_index + 1}",
        )
    x = fnx.norm(x)

    return x

def gpt2_small(x, **kwargs):
    return encode(
        x,
        depth=12,
        channels=768,
        heads=12,
        block_size=1024,
        vocab_size=50257,
        **kwargs,
    )

def gpt2_medium(x, **kwargs):
    return encode(
        x,
        depth=24,
        channels=1024,
        heads=16,
        block_size=1024,
        vocab_size=50257,
        **kwargs,
    )

def gpt2_large(x, **kwargs):
    return encode(
        x,
        depth=36,
        channels=1280,
        heads=20,
        block_size=1024,
        vocab_size=50257,
        **kwargs,
    )

def gpt2_xlarge(x, **kwargs):
    return encode(
        x,
        depth=48,
        channels=1600,
        heads=25,
        block_size=1024,
        vocab_size=50257,
        **kwargs,
    )

def gptneo_125m(x, **kwargs):
    return encode(
        x,
        depth=12,
        channels=768,
        heads=12,
        block_size=2048,
        vocab_size=50257,
        **kwargs,
    )

def gptneo_1_3b(x, **kwargs):
    return encode(
        x,
        depth=24,
        channels=2048,
        heads=16,
        block_size=2048,
        vocab_size=50257,
        **kwargs,
    )

def gptneo_2_7b(x, **kwargs):
    return encode(
        x,
        depth=32,
        channels=2560,
        heads=20,
        block_size=2048,
        vocab_size=50257,
        **kwargs,
    )
