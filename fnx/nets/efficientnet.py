import fnx, math, jax, einx
import jax.numpy as jnp

@fnx.module
def stem(x, channels):
    x = fnx.conv(x, kernel_size=3, stride=2, channels=channels, bias=False)
    x = fnx.norm(x)
    x = fnx.act(x)
    return x

@fnx.module
def block(x, channels, kernel_size, stride, expand_ratio, se_ratio=0):
    x0 = x

    if expand_ratio > 1:
        with fnx.scope("expand"):
            x = fnx.linear(x, channels=x.shape[-1] * expand_ratio, bias=False)
            x = fnx.norm(x)
            x = fnx.act(x)

    with fnx.scope("depthwise"):
        x = fnx.conv_depthwise(x, kernel_size=kernel_size, stride=stride, bias=False)
        x = fnx.norm(x)
        x = fnx.act(x)

    if se_ratio > 0:
        with fnx.scope("se"):
            se = einx.mean("b [s...] c", x)
            se = fnx.linear(se, channels=max(1, int(x0.shape[-1] * se_ratio)), bias=True)
            se = fnx.act(se)
            se = fnx.linear(se, channels=x.shape[-1], bias=True)
            se = jax.nn.sigmoid(se)
            x = einx.multiply("b s... c, b c -> b s... c", x, se, output_ndims=len(x.shape))

    with fnx.scope("out"):
        x = fnx.linear(x, channels=channels, bias=False)
        x = fnx.norm(x)

    if x.shape == x0.shape:
        x = x0 + fnx.stochastic.droppath(x)

    return x

@fnx.module
def encode(x, stem_channels, depths, channels, kernel_size, stride, expand_ratio, se_ratio):
    x = stem(x, channels=stem_channels)

    for stage_index in range(len(depths)):
        with fnx.scope(f"stage{stage_index + 1}"):
            for block_index in range(depths[stage_index]):
                x = block(
                    x,
                    channels=channels[stage_index],
                    kernel_size=kernel_size[stage_index],
                    stride=stride[stage_index] if block_index == 0 else 1,
                    expand_ratio=expand_ratio[stage_index],
                    se_ratio=se_ratio,
                    name=f"block{block_index + 1}",
                )

            if stage_index == len(depths) - 1:
                with fnx.scope("neck"):
                    x = fnx.linear(x, channels=4 * x.shape[-1], bias=False)
                    x = fnx.norm(x)
                    x = fnx.act(x)
            fnx.sow("~", x)


    return x



def round_depth(depth):
    # See: https://github.com/keras-team/keras/blob/e6784e4302c7b8cd116b74a784f4b78d60e83c26/keras/applications/efficientnet.py#L352
    return int(math.ceil(depth))

def round_channels(channels):
    # See: https://github.com/keras-team/keras/blob/e6784e4302c7b8cd116b74a784f4b78d60e83c26/keras/applications/efficientnet.py#L340
    divisor = 8
    new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)

def efficientnet_x(x, depth_coefficient, width_coefficient, **kwargs):
    return encode(
        x,
        stem_channels=round_channels(32 * width_coefficient),
        depths=[round_depth(d * depth_coefficient) for d in [1, 2, 2, 3, 3, 4, 1]],
        channels=[round_channels(c * width_coefficient) for c in [16, 24, 40, 80, 112, 192, 320]],
        kernel_size=[3, 3, 5, 3, 5, 5, 3],
        stride=[1, 2, 2, 2, 1, 2, 1],
        expand_ratio=[1, 6, 6, 6, 6, 6, 6],
        se_ratio=0.25,
        **kwargs,
    )

def efficientnet_b0(x, **kwargs):
    return efficientnet_x(
        x,
        depth_coefficient=1.0,
        width_coefficient=1.0,
        **kwargs,
    )

def efficientnet_b1(x, **kwargs):
    return efficientnet_x(
        x,
        depth_coefficient=1.1,
        width_coefficient=1.0,
        **kwargs,
    )

def efficientnet_b2(x, **kwargs):
    return efficientnet_x(
        x,
        depth_coefficient=1.2,
        width_coefficient=1.1,
        **kwargs,
    )

def efficientnet_b3(x, **kwargs):
    return efficientnet_x(
        x,
        depth_coefficient=1.4,
        width_coefficient=1.2,
        **kwargs,
    )

def efficientnet_b4(x, **kwargs):
    return efficientnet_x(
        x,
        depth_coefficient=1.8,
        width_coefficient=1.4,
        **kwargs,
    )

def efficientnet_b5(x, **kwargs):
    return efficientnet_x(
        x,
        depth_coefficient=2.2,
        width_coefficient=1.6,
        **kwargs,
    )

def efficientnet_b6(x, **kwargs):
    return efficientnet_x(
        x,
        depth_coefficient=2.6,
        width_coefficient=1.8,
        **kwargs,
    )

def efficientnet_b7(x, **kwargs):
    return efficientnet_x(
        x,
        depth_coefficient=3.1,
        width_coefficient=2.0,
        **kwargs,
    )

def efficientnet_b8(x, **kwargs):
    return efficientnet_x(
        x,
        depth_coefficient=3.6,
        width_coefficient=2.2,
        **kwargs,
    )
