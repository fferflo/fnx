import fnx, jax, einx
import jax.numpy as jnp
import numpy as np
from functools import partial
import haiku as hk

@fnx.module
def patch_embed(x, channels, patch_size, bias=True):
    x = fnx.conv(x, channels=channels, kernel_size=patch_size, stride=patch_size, bias=bias, padding=0)
    x = fnx.norm(x)
    return x

@fnx.module
def block(x, spatial_mlp_channels, channel_mlp_ratio=4):
    with fnx.scope("spatial_mix"):
        x0 = x
        x = fnx.norm(x)

        x = jnp.swapaxes(x, 1, 2)
        x = fnx.linear(x, channels=spatial_mlp_channels, bias=True, name="linear1")
        x = fnx.act(x)
        x = fnx.linear(x, channels=x0.shape[1], bias=True, name="linear2")
        x = jnp.swapaxes(x, 1, 2)

        x = fnx.layerscale(x)
        x = x0 + fnx.stochastic.droppath(x)

    with fnx.scope("channel_mix"):
        x0 = x
        x = fnx.norm(x)

        x = fnx.linear(x, channels=int(x0.shape[-1] * channel_mlp_ratio), bias=True, name="linear1")
        x = fnx.act(x)
        x = fnx.linear(x, channels=x0.shape[-1], bias=True, name="linear2")

        x = fnx.layerscale(x)
        x = x0 + fnx.stochastic.droppath(x)

    return x

@fnx.module
def encode(x, depth, channels, spatial_mlp_channels, patch_size, block=block):
    x = patch_embed(x, channels=channels, patch_size=patch_size)
    x = einx.rearrange("b s... c -> b (s...) c", x)

    for block_index in range(depth):
        x = block(x, spatial_mlp_channels, name=f"block{block_index + 1}")
    x = fnx.norm(x)

    return x



def mlp_mixer_base(x, **kwargs):
    return encode(
        x,
        depth=12,
        spatial_mlp_channels=384,
        channels=768,
        **kwargs,
    )

def mlp_mixer_large(x, **kwargs):
    return encode(
        x,
        depth=24,
        spatial_mlp_channels=512,
        channels=1024,
        **kwargs,
    )
