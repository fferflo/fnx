import fnx, jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from functools import partial

@fnx.module
def stem(x, channels):
    with fnx.scope("_1"):
        x = fnx.conv(x, channels=channels // 2, kernel_size=3, stride=2, bias=True)
        x = fnx.norm(x)
        x = fnx.act(x)
    with fnx.scope("_2"):
        x = fnx.conv(x, channels=channels, kernel_size=3, stride=2, bias=True)
        x = fnx.norm(x)
    return x

@fnx.module
def downsample(x, channels):
    x = fnx.conv(x, channels=channels, kernel_size=3, stride=2, bias=True)
    x = fnx.norm(x)
    return x

@fnx.module
def spatial_attention(x):
    x0 = x
    x = fnx.linear(x, bias=True, name="conv1")
    x = fnx.act(x)

    with fnx.scope("spatial_gating"):
        w = fnx.conv_depthwise(x, kernel_size=5, stride=1, bias=True)

        ws = [w]
        for i, k in enumerate([7, 11, 21]):
            a = fnx.conv_depthwise(w, kernel_size=(1, k), stride=1, bias=True)
            a = fnx.conv_depthwise(a, kernel_size=(k, 1), stride=1, bias=True)
            ws.append(a)
        w = sum(ws)

        w = fnx.linear(w, bias=True)

        x = w * x

    x = fnx.linear(x, bias=True, name="conv2")
    x = x + x0
    return x

@fnx.module
def block(x, channels=None, mlp_ratio=4):
    if channels is None:
        channels = x.shape[-1]

    # Self-attention
    with fnx.scope("attn"):
        x0 = x
        x = fnx.norm(x)

        x = spatial_attention(x)

        x = fnx.layerscale(x)
        x = x0 + fnx.stochastic.droppath(x)

    # MLP
    with fnx.scope("mlp"):
        x0 = x
        x = fnx.norm(x)

        x = fnx.linear(x, channels=channels * mlp_ratio, bias=True, name="pointwise1")
        x = fnx.conv_depthwise(x, kernel_size=3, stride=1, bias=True, name="depthwise")
        x = fnx.act(x)
        x = fnx.linear(x, channels=channels, bias=True, name="pointwise2")

        x = fnx.layerscale(x)
        x = x0 + fnx.stochastic.droppath(x)

    return x

@fnx.module
def encode(x, depths, channels, mlp_ratios, block=block):
    x = stem(x, channels=channels[0])

    for stage_index in range(len(depths)):
        with fnx.scope(f"stage{stage_index + 1}"):
            if stage_index > 0:
                x = downsample(x, channels=channels[stage_index])
            for block_index in range(depths[stage_index]):
                x = block(
                    x,
                    channels=channels[stage_index],
                    mlp_ratio=mlp_ratios[stage_index],
                    name=f"block{block_index + 1}",
                )
            x = fnx.norm(x)
            fnx.sow(f"~", x)

    return x



def mscan_tiny(x, block=block, **kwargs):
    return encode(
        x,
        block=block,
        depths=[3, 3, 5, 2],
        channels=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        **kwargs,
    )

def mscan_small(x, block=block, **kwargs):
    return encode(
        x,
        block=block,
        depths=[2, 2, 4, 2],
        channels=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        **kwargs,
    )

def mscan_base(x, block=block, **kwargs):
    return encode(
        x,
        block=block,
        depths=[3, 3, 12, 3],
        channels=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        **kwargs,
    )

def mscan_large(x, block=block, **kwargs):
    return encode(
        x,
        block=block,
        depths=[3, 5, 27, 3],
        channels=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        **kwargs,
    )
