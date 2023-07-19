import fnx, einx
import haiku as hk
from functools import partial
import jax.numpy as jnp

@fnx.module
def stem(x, channels):
    x = fnx.conv(x, channels=channels, kernel_size=4, stride=4, bias=True, padding=0) # TODO: bias should be false
    x = fnx.norm(x)
    return x

@fnx.module
def downsample(x, channels):
    x = fnx.norm(x)
    x = fnx.conv(x, channels=channels, kernel_size=2, stride=2, bias=True, padding=0)
    return x


# https://arxiv.org/pdf/2201.03545.pdf
@fnx.module
def block_v1(x, mlp_ratio=4):
    x0 = x

    channels = x.shape[-1]

    x = fnx.conv_depthwise(x, kernel_size=7, stride=1, bias=True, name="depthwise")
    x = fnx.norm(x)
    x = fnx.linear(x, channels=channels * mlp_ratio, bias=True, name="pointwise1")
    x = fnx.act(x)
    x = fnx.linear(x, channels=channels, bias=True, name="pointwise2")

    x = fnx.layerscale(x)
    x = x0 + fnx.stochastic.droppath(x)

    return x


# https://arxiv.org/pdf/2301.00808.pdf
@fnx.module
def global_response_normalization(x, epsilon=1e-6):
    x0 = x

    g = einx.reduce("b [s...] c", x, op=jnp.linalg.norm)
    g = einx.divide("b c, b -> b c", g, einx.mean("b [c]", g) + epsilon)
    x = einx.multiply("b s... c, b c -> b s... c", x, g)

    gamma = fnx.param("gamma", shape=[x.shape[-1]], dtype=x.dtype, init=lambda shape, dtype: jnp.zeros(shape, dtype))
    x = einx.multiply("b s... c, c -> b s... c", x, gamma)

    beta = fnx.param("beta", shape=[x.shape[-1]], dtype=x.dtype, init=lambda shape, dtype: jnp.zeros(shape, dtype))
    x = einx.add("b s... c, c -> b s... c", x, beta)

    x = x0 + x

    return x

@fnx.module
def block_v2(x, mlp_ratio=4):
    x0 = x

    channels = x.shape[-1]

    x = fnx.conv_depthwise(x, kernel_size=7, stride=1, bias=True, name="depthwise")
    x = fnx.norm(x)
    x = fnx.linear(x, channels=channels * mlp_ratio, bias=True, name="pointwise1")
    x = fnx.act(x)
    x = global_response_normalization(x)
    x = fnx.linear(x, channels=channels, bias=True, name="pointwise2")

    x = x0 + fnx.stochastic.droppath(x)

    return x


@fnx.module
def encode(x, depths, channels, block, stem=True):
    if stem:
        x = globals()["stem"](x, channels=channels[0])

    for stage_index in range(len(depths)):
        if stage_index > 0:
            x = downsample(x, channels=channels[stage_index], name=f"downsample{stage_index + 1}")

        # Blocks
        with fnx.scope(f"stage{stage_index + 1}"):
            for block_index in range(depths[stage_index]):
                x = block(x, name=f"block{block_index + 1}")
            fnx.sow("~", x)

    return x



# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py
def convnext_atto(x, **kwargs):
    return encode(
        x,
        depths=[2, 2, 6, 2],
        channels=[40, 80, 160, 320],
        **kwargs,
    )

# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py
def convnext_femto(x, **kwargs):
    return encode(
        x,
        depths=[2, 2, 6, 2],
        channels=[48, 96, 192, 384],
        **kwargs,
    )

# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py
def convnext_pico(x, **kwargs):
    return encode(
        x,
        depths=[2, 2, 6, 2],
        channels=[64, 128, 256, 512],
        **kwargs,
    )

# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py
def convnext_nano(x, **kwargs):
    return encode(
        x,
        depths=[2, 2, 8, 2],
        channels=[80, 160, 320, 640],
        **kwargs,
    )

def convnext_tiny(x, **kwargs):
    return encode(
        x,
        depths=[3, 3, 9, 3],
        channels=[96, 192, 384, 768],
        **kwargs,
    )

def convnext_small(x, **kwargs):
    return encode(
        x,
        depths=[3, 3, 27, 3],
        channels=[96, 192, 384, 768],
        **kwargs,
    )

def convnext_base(x, **kwargs):
    return encode(
        x,
        depths=[3, 3, 27, 3],
        channels=[128, 256, 512, 1024],
        **kwargs,
    )

def convnext_large(x, **kwargs):
    return encode(
        x,
        depths=[3, 3, 27, 3],
        channels=[192, 384, 768, 1536],
        **kwargs,
    )

def convnext_xlarge(x, **kwargs):
    return encode(
        x,
        depths=[3, 3, 27, 3],
        channels=[256, 512, 1024, 2048],
        **kwargs,
    )

def convnext_huge(x, **kwargs):
    return encode(
        x,
        depths=[3, 3, 27, 3],
        channels=[352, 704, 1408, 2816],
        **kwargs,
    )
