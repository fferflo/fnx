import fnx, jax
import jax.numpy as jnp
import numpy as np

@fnx.module
def interpolate_block(x, channels, bin_size, resize_method):
    input_shape = x.shape

    kernel_size = (np.asarray(input_shape[1:-1]) + bin_size - 1) // bin_size
    x = fnx.pool(x, kernel_size=kernel_size, stride=kernel_size, mode="avg", padding=0)

    x = fnx.linear(x, channels=channels, bias=False)
    x = fnx.norm(x)
    x = fnx.act(x)

    x = jax.image.resize(x, shape=input_shape[:-1] + (channels,), method=resize_method)

    return x

@fnx.module
def ppm(x, channels=None, resize_method="bilinear", bin_sizes=[6, 3, 2, 1]):
    if channels is None:
        channels = x.shape[-1] // len(bin_sizes)

    x = jnp.concatenate([x] + [interpolate_block(x, channels, bin_size, resize_method, name=f"pool{i + 1}") for i, bin_size in enumerate(bin_sizes)], axis=-1)

    return x
