import fnx, jax
import jax.numpy as jnp

@fnx.module
def decode(xs, channels, psp_bin_sizes=[1, 2, 3, 6]):
    xs = [x for x in xs]

    for i in range(len(xs) - 1):
        with fnx.scope(f"initial{i + 1}"):
            xs[i] = fnx.linear(xs[i], channels=channels, bias=False)
            xs[i] = fnx.norm(xs[i])
            xs[i] = fnx.act(xs[i])

    xs[-1] = fnx.pspnet.ppm(xs[-1], channels=channels, resize_method="bilinear", bin_sizes=psp_bin_sizes)
    with fnx.scope(f"initial{len(xs)}"):
        xs[-1] = fnx.conv(xs[-1], channels=channels, kernel_size=3, stride=1, bias=False)
        xs[-1] = fnx.norm(xs[-1])
        xs[-1] = fnx.act(xs[-1])

    for i in reversed(list(range(len(xs) - 1))):
        xs[i] = xs[i] + jax.image.resize(xs[i + 1], xs[i].shape, method="bilinear")

    for i in range(len(xs) - 1):
        with fnx.scope(f"fpn{i + 1}"):
            xs[i] = fnx.conv(xs[i], channels=channels, kernel_size=3, stride=1, bias=False)
            xs[i] = fnx.norm(xs[i])
            xs[i] = fnx.act(xs[i])

    for i in range(1, len(xs)):
        xs[i] = jax.image.resize(xs[i], xs[0].shape, method="bilinear")
    x = jnp.concatenate(xs, axis=-1)

    with fnx.scope(f"final"):
        x = fnx.conv(x, channels=channels, kernel_size=3, stride=1, bias=False)
        x = fnx.norm(x)
        x = fnx.act(x)

    return x
