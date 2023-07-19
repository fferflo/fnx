import fnx, jax, einx
import jax.numpy as jnp
import numpy as np
from functools import partial

@fnx.module
def patch_embed(x, channels, patch_size=7, stride=4, bias=True):
    x = fnx.conv(x, channels=channels, kernel_size=patch_size, stride=stride, bias=bias)
    x = fnx.norm(x)
    return x

@fnx.module
def block(x, channels=None, mlp_ratio=4, sr_ratio=1, heads=1, kernel_size=3):
    if channels is None:
        channels = x.shape[-1]

    # Self-attention
    with fnx.scope("attn"):
        # Spatial reduction attention: https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Pyramid_Vision_Transformer_A_Versatile_Backbone_for_Dense_Prediction_Without_ICCV_2021_paper.pdf
        x0 = x
        x = fnx.norm(x)

        query = fnx.linear(x, channels=1 * channels, bias=True, name="query")
        if sr_ratio > 1:
            with fnx.scope("spatial_reduction"):
                x = fnx.conv(x, channels=channels, kernel_size=sr_ratio, stride=sr_ratio, bias=True)
                x = fnx.norm(x)
        x = fnx.linear(x, channels=2 * channels, bias=True, name="keyvalue")
        key, value = jnp.split(x, indices_or_sections=2, axis=-1)
        x = fnx.attention.full_attention(
            einx.rearrange("b s... c -> b (s...) c", query),
            einx.rearrange("b s... c -> b (s...) c", key),
            einx.rearrange("b s... c -> b (s...) c", value),
            heads=heads,
        )
        x = fnx.linear(x, channels=channels, bias=True, name="out")

        x = einx.rearrange("b (s...) c -> b s... c", x, output_shape=x0.shape)

        x = x0 + fnx.stochastic.droppath(x)

    # MLP
    with fnx.scope("mlp"):
        x0 = x
        x = fnx.norm(x)

        x = fnx.linear(x, channels=channels * mlp_ratio, bias=True, name="pointwise1")
        x = fnx.conv_depthwise(x, kernel_size=kernel_size, stride=1, bias=True, name="depthwise")
        x = fnx.act(x)
        x = fnx.linear(x, channels=channels, bias=True, name="pointwise2")

        x = x0 + fnx.stochastic.droppath(x)

    return x

@fnx.module
def encode(x, depths, channels, patch_sizes, strides, sr_ratios, heads, block=block):
    for stage_index in range(len(depths)):
        with fnx.scope(f"stage{stage_index + 1}"):
            x = patch_embed(x, channels=channels[stage_index], patch_size=patch_sizes[stage_index], stride=strides[stage_index])
            for block_index in range(depths[stage_index]):
                x = block(
                    x,
                    channels=channels[stage_index],
                    sr_ratio=sr_ratios[stage_index],
                    heads=heads[stage_index],
                    name=f"block{block_index + 1}",
                )
            x = fnx.norm(x)
            fnx.sow(f"~", x)

    return x

@fnx.module
def decode(xs, channels):
    xs = [x for x in xs] # Copy list

    for i in range(len(xs)):
        xs[i] = fnx.linear(xs[i], channels=channels, bias=True, name=f"in{i + 1}")

    for i in range(1, len(xs)):
        xs[i] = jax.image.resize(xs[i], xs[0].shape, method="bilinear")
    x = jnp.concatenate(xs[::-1], axis=-1)

    with fnx.scope("fuse"):
        x = fnx.linear(x, channels=channels, bias=False)
        x = fnx.norm(x)
        x = fnx.act(x)
    return x



def mit_b0(x, block=block, **kwargs):
    return encode(
        x,
        block=block,
        depths=[2, 2, 2, 2],
        channels=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        heads=[1, 2, 5, 8],
        **kwargs,
    )

def mit_b1(x, block=block, **kwargs):
    return encode(
        x,
        block=block,
        depths=[2, 2, 2, 2],
        channels=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        heads=[1, 2, 5, 8],
        **kwargs,
    )

def mit_b2(x, block=block, **kwargs):
    return encode(
        x,
        block=block,
        depths=[3, 4, 6, 3],
        channels=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        heads=[1, 2, 5, 8],
        **kwargs,
    )

def mit_b3(x, block=block, **kwargs):
    return encode(
        x,
        block=block,
        depths=[3, 4, 18, 3],
        channels=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        heads=[1, 2, 5, 8],
        **kwargs,
    )

def mit_b4(x, block=block, **kwargs):
    return encode(
        x,
        block=block,
        depths=[3, 8, 27, 3],
        channels=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        heads=[1, 2, 5, 8],
        **kwargs,
    )

def mit_b5(x, block=block, **kwargs):
    return encode(
        x,
        block=block,
        depths=[3, 6, 40, 3],
        channels=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        heads=[1, 2, 5, 8],
        **kwargs,
    )

def segformer_b0(x, block=block):
    xs = mit_b0(x, reap=["stage1", "stage2", "stage3", "stage4"])
    x = decode(xs, channels=256)
    return x

def segformer_b1(x, block=block):
    xs = mit_b1(x, reap=["stage1", "stage2", "stage3", "stage4"])
    x = decode(xs, channels=256)
    return x

def segformer_b2(x, block=block):
    xs = mit_b2(x, reap=["stage1", "stage2", "stage3", "stage4"])
    x = decode(xs, channels=768)
    return x

def segformer_b3(x, block=block):
    xs = mit_b3(x, reap=["stage1", "stage2", "stage3", "stage4"])
    x = decode(xs, channels=768)
    return x

def segformer_b4(x, block=block):
    xs = mit_b4(x, reap=["stage1", "stage2", "stage3", "stage4"])
    x = decode(xs, channels=768)
    return x

def segformer_b5(x, block=block):
    xs = mit_b5(x, reap=["stage1", "stage2", "stage3", "stage4"])
    x = decode(xs, channels=768)
    return x
