import fnx
import numpy as np

# For more stem variants, see: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py#L482

@fnx.module(name="stem")
def stem_b(x):
    x = fnx.conv(x, channels=64, kernel_size=7, stride=2, bias=False)
    x = fnx.norm(x)
    x = fnx.act(x)
    x = fnx.pool(x, kernel_size=3, stride=2, mode="max")
    return x

@fnx.module(name="stem")
def stem_s(x, channels=[64, 64, 128], strides=[2, 1, 1]):
    for i in range(3):
        with fnx.scope(f"_{i + 1}"):
            x = fnx.conv(x, channels=channels[i], kernel_size=3, stride=strides[i], bias=False)
            x = fnx.norm(x)
            x = fnx.act(x)
    x = fnx.pool(x, kernel_size=3, stride=2, mode="max")
    return x

@fnx.module
def shortcut(x, shape, stride=1):
    if stride > 1 or x.shape[-1] != shape[-1]:
        x = fnx.conv(x, channels=shape[-1], kernel_size=1, stride=stride, bias=False)
        x = fnx.norm(x)
    return x

@fnx.module
def basic_block_v1(x, channels=None, stride=1, kernel_size=3, dilation=1):
    x0 = x

    if channels is None:
        channels = x.shape[-1]

    with fnx.scope("_1"):
        x = fnx.conv(x, channels=channels, kernel_size=kernel_size, stride=stride, bias=False, dilation=dilation)
        x = fnx.norm(x)
        x = fnx.act(x)

    with fnx.scope("_2"):
        x = fnx.conv(x, channels=channels, kernel_size=kernel_size, stride=1, bias=False)
        x = fnx.norm(x)
        x = shortcut(x0, shape=x.shape, stride=stride) + x
        x = fnx.act(x)

    return x

@fnx.module
def bottleneck_block_v1(x, channels=None, stride=1, kernel_size=3, dilation=1, factor=4):
    x0 = x

    if channels is None:
        channels = x.shape[-1]

    with fnx.scope("reduce"):
        x = fnx.linear(x, channels=channels, bias=False)
        x = fnx.norm(x)
        x = fnx.act(x)

    with fnx.scope("spatial"):
        x = fnx.conv(x, channels=channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=False)
        x = fnx.norm(x)
        x = fnx.act(x)

    with fnx.scope("expand"):
        x = fnx.linear(x, channels=channels * factor, bias=False)
        x = fnx.norm(x)
        x = shortcut(x0, shape=x.shape, stride=stride) + x
        x = fnx.act(x)

    return x

@fnx.module
def encode(x, block, depths, channels, stem=stem_b): # TODO: remove default stem
    if stem != None:
        x = stem(x)

    for stage_index in range(len(depths)):
        with fnx.scope(f"stage{stage_index + 1}"):
            for block_index in range(depths[stage_index]):
                x = block(
                    x,
                    channels=channels[stage_index],
                    stride=2 if block_index == 0 and stage_index > 0 else 1,
                    name=f"block{block_index + 1}",
                )
            fnx.sow("~", x)

    return x



def resnet_v1_18(x, **kwargs):
    return encode(x,
        block=basic_block_v1,
        depths=[2, 2, 2, 2],
        channels=[64, 128, 256, 512],
        **kwargs,
    )

def resnet_v1_34(x, **kwargs):
    return encode(x,
        block=bottleneck_block_v1,
        depths=[3, 4, 23, 3],
        channels=[64, 128, 256, 512],
        **kwargs,
    )

def resnet_v1_50(x, **kwargs):
    return encode(x,
        block=bottleneck_block_v1,
        depths=[3, 4, 6, 3],
        channels=[64, 128, 256, 512],
        **kwargs,
    )

def resnet_v1_101(x, **kwargs):
    return encode(x,
        block=bottleneck_block_v1,
        depths=[3, 4, 23, 3],
        channels=[64, 128, 256, 512],
        **kwargs,
    )

def resnet_v1_152(x, **kwargs):
    return encode(x,
        block=bottleneck_block_v1,
        depths=[3, 8, 36, 3],
        channels=[64, 128, 256, 512],
        **kwargs,
    )
