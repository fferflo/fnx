import fnx

@fnx.module
def downsample(x, channels):
    x = fnx.norm(x)

    x0 = x
    x = fnx.conv_depthwise(x, kernel_size=3, stride=1, bias=False)
    x = fnx.act(x)
    x = fnx.squeezeexcite.block(x)
    x = fnx.linear(x, bias=False)
    x = x0 + x

    x = fnx.conv(x, channels=channels, kernel_size=3, stride=2, bias=False)
    x = fnx.norm(x)

    return x

@fnx.module
def patch_embed(x, channels=96):
    x = fnx.conv(x, channels=channels, kernel_size=3, stride=2, bias=False)
    x = downsample(x, channels=channels)
    return x